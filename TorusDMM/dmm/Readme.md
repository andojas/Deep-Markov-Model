Instruction on how to use the TORUSDMM
this assumes you already have pyro installed and working

-download everything from dmm folder in github
-run the protein_data_loader.py "python protein_data_loader.py" (I forgot if it was necessary to do that before the first run, but it surely doesn't hurt)

we made a command.sh which looks like the following:

CUDA_VISIBLE_DEVICES=0 python dmm_guide.py -sopt opt3_21_02 -smod mod3_21_02 -cf 20 --cuda -n $1

CUDA_VISIBLE_DEVICES=0 : sets the cuda device
-sopt opt3_21_02 : saves the optimizer as "opt3_21_02" - change here if you want to save it with a different name 
-smod mod3_21_02 : saves the model as "mod3_21_02" - change here if you want to save it with a different name
the sopt and smod documents save the model and the optimizer and are useful to continue training from the point where you left off last time.
-cf 20 : checkpoint frequency - after this many epoches the optimizer and model will be saved, change here if you want to save it with a different frequencey
--cuda : puts the model on GPU
-n : number of epoches for which the model should be trained will be entered in the command line.

to start the model adjust the command.sh file to your needs if necessary
and type 
sh command.sh <number of epoches you want it to train for> 2>&1 | tee <name of the output file>

ELBO, RamaSample and RamaMean will be printed in the same directory.

the model will stop either after the assigned number of iterations or earlier if the elbo didnt change more than 1e-2 five times in a row. (convergence stop) the threshold is set in a variable called threshold at the beginning of the document

we exchanged the sigmoid function of kappa at the end of the emitter with a relu() + 1 function
currently we re trying different approaches for the m at the end of the emitter function

z_dim can be changed among the input parameters in __init__() in the DMM class

mini_batch_size can be changed at the end of "dmm_guide.py" by changing the default of "parser.add_argument('-mbs', '--mini-batch-size', type=int, default=150)" or by adding it as an argument in the command line when starting the model.

number of proteins can be changed in the "protein_data_loader.py" file in line 56 by reducing L_max; total number of proteins is about 402 

hope everything is understandable!
