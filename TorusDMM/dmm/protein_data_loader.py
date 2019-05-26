"""
Data loader logic with two main responsibilities:
(i)  download raw data and process; this logic is initiated upon import
(ii) helper functions for dealing with mini-batches, sequence packing, etc.

The data is data of 500 protein structures
"""

import os

import numpy as np
import six.moves.cPickle as pickle
import torch
import torch.nn as nn

import numpy.random as npr
from random import shuffle


# process actual amino_acids data
def process_data(base_path, filename):
    output = os.path.join(base_path, filename)
    # names of the aminoacids we use so we can convert them to numbers
    aminoacid_names = ['M', 'N', 'I', 'F', 'E', 'L', 'R', 'D', 'G', 'K', 'Y', 'T', 'H', 'S', 'P', 'A', 'V', 'Q', 'W', 'C']
    # open the file
    protein_file = open('top500.txt', 'r')
    print("processing raw protein data...")
    # where to store the data
    protein = []
    proteins_data = []
    # putting the data into lists
    for line in protein_file:    
        if '#' in line:
            if len(protein) > 0:
                proteins_data.append(np.array(protein))
                protein = []
        else:
            args = line.split(' ')
            if 'NT' in args:
                args[np.where(np.array(args) == 'NT')[0][0]] = npr.normal(np.pi, 0.1)
            if 'CT' in args:
                args[np.where(np.array(args) == 'CT')[0][0]] = npr.normal(np.pi, 0.1)
            # removing the sencondary structure (not used)
            args.pop(1)
            #replacing aminoacids with numbers 1-20
            args[0] = int(np.where(np.array(aminoacid_names) == args[0])[0][0])
            #changing categorical to one-hot
            onehot = np.zeros(20)
            onehot[args[0]] = 1.0
            args = np.array(args[:3]).astype(float)
            # add pi to the angles, so theyre in [0, 2pi]
            args[1] += np.pi
            args[2] += np.pi
            #saving the acids and angles
            prot = np.hstack([onehot, args[1:3]])
            protein.append(prot)
    proteins_data.append(np.array(protein))
    # the number of dimensions for protein info (amino acid name, phi, psi)
    protein_info = 22
    n_removed = 0
    # Reapply this to remove proteins of a certain length.
    # maximum length of a protein 839
    L_max = 840
    # only take proteins smaller than L_max
    
    for i in range(len(proteins_data)):
        i -= n_removed
        if len(proteins_data[i]) > L_max:
            proteins_data.pop(i)
            n_removed += 1
    # randomized order of the proteins so no bias
    shuffle(proteins_data)
    # use the same percentage as in jsb_chorales, 60 20 20
    p_60, p_80 = np.int(0.95*len(proteins_data)), np.int(0.98*len(proteins_data))
    proteins_data = [proteins_data[:p_60], proteins_data[p_60:p_80], proteins_data[p_80:]]
    # This is the shape we need
    processed_dataset = {}
    for split, data_split in zip(['train', 'test', 'valid'], proteins_data):
            processed_dataset[split] = {}
            n_seqs = len(data_split)
            processed_dataset[split]['sequence_lengths'] = np.zeros((n_seqs), dtype=np.int32)
            processed_dataset[split]['sequences'] = np.zeros((n_seqs, L_max, protein_info))
            for seq in range(n_seqs):
                seq_length = len(data_split[seq])
                #print(seq)
                processed_dataset[split]['sequence_lengths'][seq] = seq_length
                for l in range(seq_length):
                    protein_slice = np.array(data_split[seq][l])
                    processed_dataset[split]['sequences'][seq, l, :] = protein_slice

    pickle.dump(processed_dataset, open(output, "wb"), pickle.HIGHEST_PROTOCOL)
    print("dumped processed data to %s" % output)
    #print(processed_dataset)


# this logic will be initiated upon import
base_path = './data'# get_data_directory(__file__) removed becaue annoying
process_data(base_path, "proteins_processed.pkl")
protein_file_loc = os.path.join(base_path, "proteins_processed.pkl")


# ingest training/validation/test data from disk
def load_data():
    with open(protein_file_loc, "rb") as f:
        return pickle.load(f)


# this function takes a numpy mini-batch and reverses each sequence
# (w.r.t. the temporal axis, i.e. axis=1)
def reverse_sequences_numpy(mini_batch, seq_lengths):
    reversed_mini_batch = mini_batch.copy()
    for b in range(mini_batch.shape[0]):
        T = seq_lengths[b]
        reversed_mini_batch[b, 0:T, :] = mini_batch[b, (T - 1)::-1, :]
    return reversed_mini_batch


# this function takes a torch mini-batch and reverses each sequence
# (w.r.t. the temporal axis, i.e. axis=1)
# in contrast to `reverse_sequences_numpy`, this function plays
# nice with torch autograd
def reverse_sequences_torch(mini_batch, seq_lengths):
    reversed_mini_batch = mini_batch.new_zeros(mini_batch.size())
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = np.arange(T - 1, -1, -1)
        time_slice = torch.cuda.LongTensor(time_slice) if 'cuda' in mini_batch.data.type() \
            else torch.LongTensor(time_slice)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch


# this function takes the hidden state as output by the PyTorch rnn and
# unpacks it it; it also reverses each sequence temporally
def pad_and_reverse(rnn_output, seq_lengths):
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_sequences_torch(rnn_output, seq_lengths)
    return reversed_output


# this function returns a 0/1 mask that can be used to mask out a mini-batch
# composed of sequences of length `seq_lengths`
def get_mini_batch_mask(mini_batch, seq_lengths):
    mask = np.zeros(mini_batch.shape[0:2])
    for b in range(mini_batch.shape[0]):
        mask[b, 0:seq_lengths[b]] = np.ones(seq_lengths[b])
    return mask


# this function prepares a mini-batch for training or evaluation.
# it returns a mini-batch in forward temporal order (`mini_batch`) as
# well as a mini-batch in reverse temporal order (`mini_batch_reversed`).
# it also deals with the fact that packed sequences (which are what what we
# feed to the PyTorch rnn) need to be sorted by sequence length.
def get_mini_batch(mini_batch_indices, sequences, seq_lengths, cuda=False):
    # get the sequence lengths of the mini-batch
    seq_lengths = seq_lengths[mini_batch_indices]
    # sort the sequence lengths
    sorted_seq_length_indices = np.argsort(seq_lengths)[::-1]
    sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
    sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]
    # compute the length of the longest sequence in the mini-batch
    T_max = np.max(seq_lengths)
    # this is the sorted mini-batch
    mini_batch = sequences[sorted_mini_batch_indices, 0:T_max, :]
    # this is the sorted mini-batch in reverse temporal order
    mini_batch_reversed = reverse_sequences_numpy(mini_batch, sorted_seq_lengths)
    # get mask for mini-batch
    mini_batch_mask = get_mini_batch_mask(mini_batch, sorted_seq_lengths)

    # wrap in PyTorch Tensors, using default tensor type
    mini_batch = torch.tensor(mini_batch).type(torch.Tensor)
    mini_batch_reversed = torch.tensor(mini_batch_reversed).type(torch.Tensor)
    mini_batch_mask = torch.tensor(mini_batch_mask).type(torch.Tensor)

    # cuda() here because need to cuda() before packing
    if cuda:
        mini_batch = mini_batch.cuda()
        mini_batch_mask = mini_batch_mask.cuda()
        mini_batch_reversed = mini_batch_reversed.cuda()

    # do sequence packing
    mini_batch_reversed = nn.utils.rnn.pack_padded_sequence(mini_batch_reversed,
                                                            sorted_seq_lengths,
                                                            batch_first=True)
    #print("protein_data_loader", mini_batch[:,:,0:20].sum(2))
    return mini_batch, mini_batch_reversed, mini_batch_mask, sorted_seq_lengths
