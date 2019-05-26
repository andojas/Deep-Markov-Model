# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import glob
import os.path
import os
import platform
import numpy as np
import h5py
from util import AA_ID_DICT, calculate_dihedral_angles, protein_id_to_str, get_structure_from_angles, \
    structure_to_backbone_atoms, write_to_pdb, calculate_dihedral_angles_over_minibatch, \
    get_backbone_positions_from_angular_prediction, encode_primary_string
import torch

MAX_SEQUENCE_LENGTH = 2000

def process_raw_data(use_gpu, force_pre_processing_overwrite=True):
    print("Starting pre-processing of raw data...")
    input_files = glob.glob("data/raw/*")   # adds all files in that directory to a list
    print(input_files)
    input_files_filtered = filter_input_files(input_files)  # filters input files according to the filter_input_files function specified at the bottom of the script
    for file_path in input_files_filtered:  # iterates over filtered input files
        if platform.system() is 'Windows':  
            filename = file_path.split('\\')[-1]    # gets the name of the input file
        else:
            filename = file_path.split('/')[-1]     # gets the name of the input file
        preprocessed_file_name = "data/preprocessed/"+filename+".hdf5"  # creates a filename for a preprocessed input file

        # check if we should remove the any previously processed files
        if os.path.isfile(preprocessed_file_name):  # this statement is correct if there is already a file with the same name in the directory
            print("Preprocessed file for " + filename + " already exists.")
            if force_pre_processing_overwrite:
                print("force_pre_processing_overwrite flag set to True, overwriting old file...")
                os.remove(preprocessed_file_name)   # if force_pre_processing_overwrite is set to True it removes the previous file
            else:
                print("Skipping pre-processing for this file...")

        if not os.path.isfile(preprocessed_file_name):
            process_file(filename, preprocessed_file_name, use_gpu)
    print("Completed pre-processing.")

def read_protein_from_file(file_pointer):

        dict_ = {}
        _dssp_dict = {'L': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7}
        _mask_dict = {'-': 0, '+': 1}

        while True:
            next_line = file_pointer.readline()
            if next_line == '[ID]\n':   # if the line in the file contains its ID include it in the dictionary under the id key
                id_ = file_pointer.readline()[:-1] # such specific indexing is used to omit the last character in this case the newline
                dict_.update({'id': id_})
            elif next_line == '[PRIMARY]\n':    # if the line in the file contains its Primary structure info include it in the dictionary under the primary key
                primary = encode_primary_string(file_pointer.readline()[:-1])   # creates an encoded list where each aa is changed to its alphabetical position among all aa
                dict_.update({'primary': primary})
            elif next_line == '[EVOLUTIONARY]\n':
                evolutionary = []
                for residue in range(21): evolutionary.append(
                    [float(step) for step in file_pointer.readline().split()])  # the line contains 21 times the seq_length=lenseq of numbers first lenseq amount corresponds to the first aa the next lenseq to the second one and etc. in the end it appends 21 lists to a list which is then included in the dictionary under the key evolutionary
                dict_.update({'evolutionary': evolutionary})
            elif next_line == '[SECONDARY]\n':  # if the line in the file contains its Secondary structure info include it in the dictionary under the secondary key
                secondary = list([_dssp_dict[dssp] for dssp in file_pointer.readline()[:-1]])
                dict_.update({'secondary': secondary})
            elif next_line == '[TERTIARY]\n':   # if the line in the file contains its Tertiary structure info include it in the dictionary under the tertiary key
                tertiary = []
                # 3 dimension
                for axis in range(3): tertiary.append(  # first appends all first backbone atom coordinates of every aa, then the second and finally the last one
                    [float(coord) for coord in file_pointer.readline().split()])
                dict_.update({'tertiary': tertiary})
            elif next_line == '[MASK]\n':
                mask = list([_mask_dict[aa] for aa in file_pointer.readline()[:-1]])
                dict_.update({'mask': mask})
            elif next_line == '\n':
                return dict_
            elif next_line == '':
                return None


def process_file(input_file, output_file, use_gpu):
    print("Processing raw data file", input_file)

    # create output file
    f = h5py.File(output_file, 'w')
    current_buffer_size = 1
    current_buffer_allocation = 0
    dset1 = f.create_dataset('primary',(current_buffer_size,MAX_SEQUENCE_LENGTH),maxshape=(None,MAX_SEQUENCE_LENGTH),dtype='int32') # creates an empty dataset with given dimension, axes with none in maxshape are unlimited
    dset2 = f.create_dataset('tertiary',(current_buffer_size,MAX_SEQUENCE_LENGTH,9),maxshape=(None,MAX_SEQUENCE_LENGTH, 9),dtype='float')
    dset3 = f.create_dataset('mask',(current_buffer_size,MAX_SEQUENCE_LENGTH),maxshape=(None,MAX_SEQUENCE_LENGTH),dtype='uint8')

    input_file_pointer = open("data/raw/" + input_file, "r")

    while True:
        # while there's more proteins to process
        next_protein = read_protein_from_file(input_file_pointer)
        if next_protein is None:
            break

        sequence_length = len(next_protein['primary'])

        if sequence_length > MAX_SEQUENCE_LENGTH:
            print("Dropping protein as length too long:", sequence_length)
            continue

        if current_buffer_allocation >= current_buffer_size:
            current_buffer_size = current_buffer_size + 1
            dset1.resize((current_buffer_size,MAX_SEQUENCE_LENGTH))
            dset2.resize((current_buffer_size,MAX_SEQUENCE_LENGTH, 9))
            dset3.resize((current_buffer_size,MAX_SEQUENCE_LENGTH))

        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        tertiary_padded = np.zeros((9, MAX_SEQUENCE_LENGTH))
        mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)

        # masking and padding here happens so that the stored dataset is of the same size. 
        # when the data is loaded in this padding is removed again. 
        primary_padded[:sequence_length] = next_protein['primary']
        t_transposed = np.ravel(np.array(next_protein['tertiary']).T)   # flattens the array into 1-D
        t_reshaped = np.reshape(t_transposed, (sequence_length,9)).T

        tertiary_padded[:,:sequence_length] = t_reshaped
        mask_padded[:sequence_length] = next_protein['mask']

        mask = torch.Tensor(mask_padded).type(dtype=torch.uint8)
        
        prim = torch.masked_select(torch.Tensor(primary_padded).type(dtype=torch.long), mask)   # only leaves those aa which have + (actually 0) in their mask
        pos = torch.masked_select(torch.Tensor(tertiary_padded), mask).view(9, -1).transpose(0, 1).unsqueeze(1) / 100   # divides by 100 because all values are artificially increased by 100 as specified in the proteinnet documentation, do not know yet why we need to add an additional dimension???

        if use_gpu:
            pos = pos.cuda()

        angles, batch_sizes = calculate_dihedral_angles_over_minibatch(pos, [len(prim)], use_gpu=use_gpu)

        tertiary, _ = get_backbone_positions_from_angular_prediction(angles, batch_sizes, use_gpu=use_gpu)
        tertiary = tertiary.squeeze(1)

        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        tertiary_padded = np.zeros((MAX_SEQUENCE_LENGTH, 9))

        length_after_mask_removed = len(prim)

        primary_padded[:length_after_mask_removed] = prim.data.cpu().numpy()
        tertiary_padded[:length_after_mask_removed, :] = tertiary.data.cpu().numpy()
        mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        mask_padded[:length_after_mask_removed] = np.ones(length_after_mask_removed)

        dset1[current_buffer_allocation] = primary_padded
        dset2[current_buffer_allocation] = tertiary_padded
        dset3[current_buffer_allocation] = mask_padded
        current_buffer_allocation += 1

    print("Wrote output to", current_buffer_allocation, "proteins to", output_file)


def filter_input_files(input_files):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    return list(filter(lambda x: not x.endswith(disallowed_file_endings), input_files))