"""
An implementation of a Deep Markov Model in Pyro based on reference [1].
This is essentially the DKS variant outlined in the paper. The primary difference
between this implementation and theirs is that in our version any KL divergence terms
in the ELBO are estimated via sampling, while they make use of the analytic formulae.
We also illustrate the use of normalizing flows in the variational distribution (in which
case analytic formulae for the KL divergences are in any case unavailable).
Reference:
[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag
"""

import argparse
import time
from os.path import exists

import numpy as np
import numpy.random as npr
import six.moves.cPickle as pickle
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors



import protein_data_loader as poly
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import InverseAutoregressiveFlow, TransformedDistribution
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, Importance, EmpiricalMarginal
from pyro.nn import AutoRegressiveNN
from pyro.optim import ClippedAdam
from util import get_logger 
from pprint import pprint

# For the scales in combiner and transition
MIN_VAR = 1e-3
#global variable for threshold
threshold = 1e-4

#print("kappa: * 99 + 1")

class Emitter(nn.Module):
    """
    Parameterizes observation likelihood `p(x_t | z_t)`
    """
    # i removed input_dim because its not needed here
    def __init__(self, z_dim, emission_dim, amino_acid_dim=20):
        super(Emitter, self).__init__()
        # initialize the three linear transformations used in the neural network
        self.amino_acid_dim = amino_acid_dim
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        # this is for the amino acid
        self.lin_hidden_to_output_acids = nn.Linear(emission_dim, amino_acid_dim)
        
        # these are for the angles phi and psi, for kappa and my
        self.lin_hidden_to_output_phi_k = nn.Linear(emission_dim, 1)
        self.lin_hidden_to_output_phi_m = nn.Linear(emission_dim, 1)
        self.lin_hidden_to_output_psi_k = nn.Linear(emission_dim, 1)
        self.lin_hidden_to_output_psi_m = nn.Linear(emission_dim, 1)

        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU() #rectfied linear unit
        self.softmax = nn.Softmax(dim=-1) #for the amino acids
        self.sigmoid = nn.Sigmoid() # for the angles
        self.softplus = nn.Softplus()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the Bernoulli distribution `p(x_t|z_t)`
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        h3 = self.relu(self.lin_hidden_to_hidden(h2))
        h4 = self.relu(self.lin_hidden_to_hidden(h3))
        # acids nn to output
        ps = self.lin_hidden_to_output_acids(h3)
        # reshaping to (:, 1, 20)
        ps = ps.view(-1, 1, self.amino_acid_dim)
        # softmax for the categorical
        ps = self.softmax(ps)

        # angles hidden layer to output 
        ps_phi_k = self.lin_hidden_to_output_phi_k(h4)
        ps_phi_m = self.lin_hidden_to_output_phi_m(h4) #dim 50,2
        ps_psi_k = self.lin_hidden_to_output_psi_k(h4)
        ps_psi_m = self.lin_hidden_to_output_psi_m(h4)
        
        # applying the sigmoid and relu
        # times 2 pi for the mu's
        # unlimited kappas with relu+1 function
        ps_phi_k = self.softplus(ps_phi_k)
        ps_psi_k = self.softplus(ps_psi_k)
        #from IPython.core.debugger import set_trace
        #set_trace()
        ps_phi_m = self.sigmoid(ps_phi_m)*2*np.pi
        ps_psi_m = self.sigmoid(ps_psi_m)*2*np.pi
        # here i added 4 extra returns, so i also changed it in the model
        return ps, ps_phi_k, ps_phi_m, ps_psi_k, ps_psi_m


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """
    def __init__(self, z_dim, transition_dim):
        super(GatedTransition, self).__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # add small constant to not get zeroes
        scale = scale + MIN_VAR
        # return loc, scale which can be fed into Normal
        return loc, scale


class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """
    def __init__(self, z_dim, rnn_dim):
        super(Combiner, self).__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        # add small constant to not get zeroes
        scale = scale + MIN_VAR
        # return loc, scale which can be fed into Normal
        return loc, scale


class DMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """
    # the input dim has to match the data, for proteins = 22, aminoacid nr = 20 and 2 angles
    #z_dim set to 5 instead of 100
    def __init__(self, input_dim=22, z_dim=100, emission_dim=100,
                 transition_dim=200, rnn_dim=600, rnn_dropout_rate=0.0,
                 num_iafs=0, iaf_dim=200, use_cuda=False):
        print('z_dim:', z_dim)
        super(DMM, self).__init__()
        # instantiate PyTorch modules used in the model and guide below
        self.emitter = Emitter(z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                          batch_first=True, bidirectional=False, num_layers=1,
                          dropout=rnn_dropout_rate)

        # if we're using normalizing flows, instantiate those too
        self.iafs = [InverseAutoregressiveFlow(AutoRegressiveNN(z_dim, [iaf_dim])) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()
        else:
            self.cpu()

    # the model p(x_{1:T} | z_{1:T}) p(z_{1:T})
    #changed to None because for the sampling we dont have observed data
    def model(self, mini_batch=None, mini_batch_reversed=None, mini_batch_mask=None,
              mini_batch_seq_lengths=None, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        # we still have mini_batch_mask, even for the sampling
        T_max = mini_batch.size(1) if mini_batch is not None else mini_batch_mask.size(1)
        #print(mini_batch.size())
        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(mini_batch_mask.size(0), self.z_0.size(0))

        # we enclose all the sample statements in the model in a iarange.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(mini_batch_mask)):
            # sample the latents z and observed x's one time step at a time
            for t in range(1, T_max + 1):
                # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
                # note that (both here and elsewhere) we use poutine.scale to take care
                # of KL annealing. we use the mask() method to deal with raggedness
                # in the observed data (i.e. different sequences in the mini-batch
                # have different lengths)

                # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.trans(z_prev)

                # then sample z_t according to dist.Normal(z_loc, z_scale)
                # note that we use the reshape method so that the univariate Normal distribution
                # is treated as a multivariate Normal distribution with a diagonal covariance.
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t,
                                      dist.Normal(z_loc, z_scale)
                                          .mask(mini_batch_mask[:, t - 1:t])
                                          .to_event(1))

                # compute the probabilities that parameterize likelihoods
                # for OneHotcategorical, and von mises. em stands for emission
                em_probs_acids, em_probs_phi_k, em_probs_phi_m, em_probs_psi_k, em_probs_psi_m = self.emitter(z_t)
                # the next statement instructs pyro to observe x_t according to the
                # OneHotcategorical distribution p(x_t|z_t)
                if mini_batch is not None:
                    observed_categorical = mini_batch[:, t - 1, 0:20]
                    observed_categorical_zero = (observed_categorical.sum(-1) == 0).nonzero()
                    dummy_onehot = observed_categorical.new_zeros(20)
                    dummy_onehot[0] = 1
                    observed_categorical[observed_categorical_zero] = dummy_onehot
                    pyro.sample("obs_x_%d" % t,
                                dist.OneHotCategorical(em_probs_acids)
                                    .mask(mini_batch_mask[:, t - 1:t])
                                    .to_event(1),
                                obs= observed_categorical)

                    #print(mini_batch[:,t-1,0:20].sum(-1))
                    #print(mini_batch[-1, t-1, 0:20])
                    # here put in sampling from von mises
                    # using mu and kappa for the angles
                    pyro.sample("obs_phi_%d" % t, 
                                dist.VonMises(em_probs_phi_m, em_probs_phi_k)
                                            .mask(mini_batch_mask[:, t - 1:t])
                                            .to_event(1),
                                        obs=mini_batch[:, t - 1, 20:21])
                    pyro.sample("obs_psi_%d" % t, 
                                dist.VonMises(em_probs_psi_m, em_probs_psi_k)
                                            .mask(mini_batch_mask[:, t - 1:t])
                                            .to_event(1),
                                        obs=mini_batch[:, t - 1, 21:22])
                

                else:
                    # the same but without observed
                    pyro.sample("x_%d" % t,
                                dist.OneHotCategorical(em_probs_acids)
                                    .mask(mini_batch_mask[:, t - 1:t])
                                    .to_event(1))                    

                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t

    # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    # Set all to None and changed all mini_batch to mini_batch_mask
    def guide(self, mini_batch, mini_batch_reversed, mini_batch_mask,
              mini_batch_seq_lengths, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1) if mini_batch is not None else mini_batch_mask.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(1, mini_batch_mask.size(0), self.rnn.hidden_size).contiguous()
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(mini_batch_mask.size(0), self.z_q_0.size(0))

        # we enclose all the sample statements in the guide in a iarange.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(mini_batch_mask)):
            # sample the latents z one time step at a time
            for t in range(1, T_max + 1):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                # if we are using normalizing flows, we apply the sequence of transformations
                # parameterized by self.iafs to the base distribution defined in the previous line
                # to yield a transformed distribution that we use for q(z_t|...)
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                else:
                    z_dist = dist.Normal(z_loc, z_scale)
                assert z_dist.event_shape == ()
                assert z_dist.batch_shape == (len(mini_batch_mask), self.z_q_0.size(0))

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t,
                                      z_dist.mask(mini_batch_mask[:, t - 1:t])
                                            .to_event(1))
                # the latent sampled at this time step will be conditioned upon in the next time step
                # so keep track of it
                z_prev = z_t
    
    #Samples from the aminoacids
    def sample(self, svi, seq_lengths, num_samples=1):
        # defines the posterior
        posterior = Importance(self.model, num_samples=num_samples)
        #svi.num_samples=num_samples
        # We need the mini_batch_mask for the distributions
        mask = torch.arange(seq_lengths.float().max()) < seq_lengths.float().unsqueeze(-1)
        # Makes the samples
        trace = posterior.run(mini_batch_mask=mask.float())

        # calculate estimate of parameters using the trace
        marginal_x = trace.marginal(["x_{}".format(i) for i in range(1,seq_lengths.max()+1)]) #comprehension  
        # estimate the z's using the trace
        marginal_z = trace.marginal(["z_{}".format(i) for i in range(1,seq_lengths.max()+1)]) #comprehension 
        #print(trace.marginal(["z_{}".format(i) for i in range(1,seq_lengths.max()+1)]).support())
        
        marginal_x_sup = marginal_x.support()
        marginal_z_sup = marginal_z.support()
        z_list = []
        x_list = []

        for k in range(1,seq_lengths.max()+1):
            z_list.append(marginal_z_sup["z_{}".format(k)])
            x_list.append(marginal_x_sup["x_{}".format(k)])

        marginal_z_samples = torch.stack(z_list, dim=2)
        marginal_x_samples = torch.stack(x_list, dim=2)
        print(marginal_z_samples.size())

        outputs_amino = []
        outputs_angles = []
        # returns the samples as the marginal mean
        #loop over proteins       
        angles_means = []
        angles_ks = []
        for i,length in enumerate (seq_lengths):
            output_amino = []
            output_angles = []
            angle_mean = []
            angle_k = []
            #loop over length of protein
            for j in range(length):
                output_amino.append(marginal_x_samples[0, i, j].argmax())
                # find the mean and kappa of each angle for each amino acid
                ps, ps_phi_k, ps_phi_m, ps_psi_k, ps_psi_m = self.emitter(marginal_z_samples[0:1, i, j])
                # Draw samples using numpy
                phi = npr.vonmises(ps_phi_m.detach()-np.pi, ps_phi_k.detach())+ np.pi
                psi = npr.vonmises(ps_psi_m.detach()-np.pi, ps_psi_k.detach())+ np.pi 
                # save the sampled angles to use in ramachandran plot
                angle_mean.append([ps_phi_m.detach()[0], ps_psi_m.detach()[0]])
                angle_k.append([ps_phi_k.detach()[0], ps_psi_k.detach()[0]])
                output_angles.append([phi[0], psi[0]])
            outputs_amino.append(output_amino)
            outputs_angles.append(output_angles)
            angles_means.append(angle_mean)
            angles_ks.append(angle_k)

        return np.array(outputs_amino), np.array(outputs_angles), np.array(angles_means), np.array(angles_ks)

# setup, training, and evaluation
def main(args):
    # setup logging
    log = get_logger(args.log)
    log(args)

    # ingest training/validation/test data from disk
    data = poly.load_data()
    training_seq_lengths = data['train']['sequence_lengths']
    training_data_sequences = data['train']['sequences']
    test_seq_lengths = data['test']['sequence_lengths']
    test_data_sequences = data['test']['sequences']
    val_seq_lengths = data['valid']['sequence_lengths']
    val_data_sequences = data['valid']['sequences']
    N_train_data = len(training_seq_lengths)
    N_train_time_slices = float(np.sum(training_seq_lengths))
    N_mini_batches = int(N_train_data / args.mini_batch_size +
                         int(N_train_data % args.mini_batch_size > 0))

    log("N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d" %
        (N_train_data, np.mean(training_seq_lengths), N_mini_batches))
    
    # instantiate the dmm
    dmm = DMM(rnn_dropout_rate=args.rnn_dropout_rate, num_iafs=args.num_iafs,
              iaf_dim=args.iaf_dim, use_cuda=args.cuda)

    # setup optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta1, args.beta2),
                   "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                   "weight_decay": args.weight_decay}
    adam = ClippedAdam(adam_params)

    # setup inference algorithm
    elbo = JitTrace_ELBO(num_particles=5) if args.jit else Trace_ELBO(num_particles=5)
    svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)

    # now we're going to define some functions we need to form the main training loop

    # saves the model and optimizer states to disk
    def save_checkpoint(epoch_list):
        log("saving model to %s..." % args.save_model)
        torch.save(dmm.state_dict(), args.save_model)
        log("saving optimizer states to %s..." % args.save_opt)
        adam.save(args.save_opt)
        log("done saving model and optimizer checkpoints to disk.")
        np.savetxt("epoch_list.txt", np.array(epoch_list))

    # loads the model and optimizer states from disk
    def load_checkpoint():
        assert exists(args.load_opt) and exists(args.load_model), \
            "--load-model and/or --load-opt misspecified"
        log("loading model from %s..." % args.load_model)
        dmm.load_state_dict(torch.load(args.load_model))
        log("loading optimizer states from %s..." % args.load_opt)
        adam.load(args.load_opt)
        log("done loading model and optimizer states.")
        epoch_list=list(np.genfromtxt("epoch_list.txt"))
        return (epoch_list)

    # prepare a mini-batch and take a gradient step to minimize -elbo
    def process_minibatch(epoch, which_mini_batch, shuffled_indices):
        if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
            # compute the KL annealing factor approriate for the current mini-batch in the current epoch
            min_af = args.minimum_annealing_factor
            annealing_factor = min_af + (1.0 - min_af) * \
                (float(which_mini_batch + epoch * N_mini_batches + 1) /
                 float(args.annealing_epochs * N_mini_batches))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        # compute which sequences in the training set we should grab
        mini_batch_start = (which_mini_batch * args.mini_batch_size)
        mini_batch_end = np.min([(which_mini_batch + 1) * args.mini_batch_size, N_train_data])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
        # grab a fully prepped mini-batch using the helper function in the data loader
        mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
            = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
                                  training_seq_lengths, cuda=args.cuda)
        #print(mini_batch[:,:,0:20].sum(2))
        # do an actual gradient step
        loss = svi.step(mini_batch, mini_batch_reversed, mini_batch_mask,
                        mini_batch_seq_lengths, annealing_factor)
        # keep track of the training loss
        return loss
    
    # if checkpoint files provided, load model and optimizer states from disk before we start training
    if args.load_opt != '' and args.load_model != '':
        epoch_list=load_checkpoint()

    def make_plots(epoch_list, training_seq_lengths, elbo):
    	# plot the epochs
        # its nice to have the training converging
        n_iter = len(epoch_list)
        x_train = np.array(range(len(epoch_list)))
        plt.title('- ELBO')
        plt.xlabel('number of epochs')
        plt.plot(x_train, epoch_list)
        #elbo_fn = "ELBO_" + str(n_iter) + ".png"
        plt.savefig("-ELBO.png")
        # Run the sampling
        # Sample same set as the original data
        training_shape = torch.tensor(training_seq_lengths, dtype=torch.float)
        dmm.cpu()
        print("Sampling ",len(training_seq_lengths), "proteins of average length ", np.mean(training_seq_lengths), "...")
        samples_amino, samples_angles, mean_angles, k_angles = dmm.sample(elbo, training_shape)
    
        print("Making plots...")

        s_phi = []
        s_psi = []
        m_phi = []
        m_psi = []
        k_phi = []
        k_psi = []
    
        for i in range(len(samples_angles)):
        	#print(samples_angles[i])
            s_i = np.array(samples_angles[i])
            m_i = np.array(mean_angles[i])
            k_i = np.array(k_angles[i])
            s_phi = s_phi + s_i[:, 0].flatten().tolist()
            s_phi_plot = np.asarray(s_phi)
            s_phi_plot = (s_phi_plot - np.pi)*(180/np.pi)

            s_psi = s_psi + s_i[:, 1].flatten().tolist()
            s_psi_plot = np.asarray(s_psi)
            s_psi_plot = (s_psi_plot - np.pi)*(180/np.pi)

            m_phi = m_phi + m_i[:, 0].flatten().tolist()
            m_phi_plot = np.asarray(m_phi)
            m_phi_plot = (m_phi_plot - np.pi)*(180/np.pi)

            m_psi = m_psi + m_i[:, 1].flatten().tolist()
            m_psi_plot = np.asarray(m_psi)
            m_psi_plot = (m_psi_plot - np.pi)*(180/np.pi)

            k_phi = k_phi + k_i[:, 0].flatten().tolist()
            k_psi = k_psi + k_i[:, 1].flatten().tolist()

        n_bins = 60
        axes = [[-180, 180], [-180, 180]]
    	#make ramachandran plot
    	# clearf figure, make plots

       ##-## plot sampled angles RAMA
        plt.clf()
        plt.xlabel('phi')
        plt.ylabel('psi')
        plt.hist2d(s_phi_plot, s_psi_plot, bins=n_bins, range=axes, norm=colors.LogNorm())
        locs, labels = plt.xticks()
        plt.xticks((-180, 0, 180))
        locs, labels = plt.yticks()
        plt.yticks((-180, 0, 180))
        RamSam_fn = "RamaSample_" + str(n_iter) +".png"
        plt.savefig(RamSam_fn)
        
        ##-## plot means
        plt.clf()
        plt.xlabel('phi')
        plt.ylabel('psi')
        plt.hist2d(m_phi_plot, m_psi_plot, bins=n_bins, range=axes, norm=colors.LogNorm())
        locs, labels = plt.xticks()
        plt.xticks((-180, 0, 180))
        locs, labels = plt.yticks()
        plt.yticks((-180, 0, 180))
        Mean_fn = "Means_" + str(n_iter) +".png"
        plt.savefig(Mean_fn)
        
        ##-## plot kappasplt.clf()
        plt.xlabel('phi')
        plt.ylabel('psi')
        plt.hist2d(k_phi, k_psi, bins=n_bins, norm=colors.LogNorm())
        k_fn = "Kappas_" + str(n_iter) +".png"
        plt.savefig(k_fn)

        plt.clf()
        dmm.cuda()

        return

    #################
    # TRAINING LOOP #
    #################
    times = [time.time()]
    try:
        len(epoch_list)
        print("read in epoch_list")
    except:
        epoch_list = []
        print("creates new epoch_list")
    x=0
    for epoch in range(args.num_epochs):
        # if specified, save model and optimizer states to disk every checkpoint_freq epochs
        if args.checkpoint_freq > 0 and epoch > 0 and epoch % args.checkpoint_freq == 0:
            save_checkpoint(epoch_list)
            make_plots(epoch_list, training_seq_lengths, elbo)

        # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
        epoch_nll = 0.0
        # prepare mini-batch subsampling indices for this epoch
        shuffled_indices = np.arange(N_train_data)
        np.random.shuffle(shuffled_indices)

        

        # process each mini-batch; this is where we take gradient steps
        for which_mini_batch in range(N_mini_batches):
            epoch_nll += process_minibatch(epoch, which_mini_batch, shuffled_indices)
        #epoch_list.append(epoch_nll)

        # report training diagnostics
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        log("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" %
            (epoch, epoch_nll / N_train_time_slices, epoch_time))
        epoch_list.append(epoch_nll / N_train_time_slices)
        
        #check for convergence
        if epoch > 0:
            if abs(epoch_list[epoch-1] - epoch_list[epoch]) < threshold:
                x += 1
                if x == 5:
                    break
            else:
                x = 0

    

# parse command-line arguments and execute the main method
if __name__ == '__main__':
    #assert pyro.__version__.startswith('0.3.0')

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001)
    parser.add_argument('-b1', '--beta1', type=float, default=0.96)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=20.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996)
    parser.add_argument('-wd', '--weight-decay', type=float, default=2.0)
    parser.add_argument('-mbs', '--mini-batch-size', type=int, default=200)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=200)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.1)
    parser.add_argument('-rdr', '--rnn-dropout-rate', type=float, default=0.1)
    parser.add_argument('-iafs', '--num-iafs', type=int, default=1)
    parser.add_argument('-id', '--iaf-dim', type=int, default=200)
    parser.add_argument('-cf', '--checkpoint-freq', type=int, default=50)
    parser.add_argument('-lopt', '--load-opt', type=str, default='')
    parser.add_argument('-lmod', '--load-model', type=str, default='')
    parser.add_argument('-sopt', '--save-opt', type=str, default='')
    parser.add_argument('-smod', '--save-model', type=str, default='')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    args = parser.parse_args()

    main(args)
