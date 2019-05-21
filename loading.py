import pandas as pd
import numpy as np
import random
import os
import fnmatch

def load_labels(labels_file_name):
    pd.read_csv(labels_file_name)

def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


# ATTENTION : in the current configuration, everytime this function is called it goes through
# all the sub-directories even if the sample is set to 6
# it only prunes the output at the end
# > there could be a more efficient way for doing this
# TODO later
def find_files(directory, pattern='*.mp3', sample=None, sub_dir=None):
    '''Recursively finds all files matching the pattern.'''
    # subdir sould be a string, for example "abc03",
    # meaning we take data from directories a,b,c,0 and 3

    files = []
    directories = []

    # TODO : add lines to check format of input
    # TODO : try/except ?

    if sub_dir!=None :
        for c in sub_dir :
            directories.append(directory + c + "/")
    else :
        directories.append(directory)

    for path in directories :
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))

    if sample!=None :
        try:
            return files[:sample]
        except TypeError:
            print("Argument sample should be either None, or an integer :\
             the number of first n samples to take.")
    else :
        return files


def find_files_group(directory, group_size, pattern='*.mp3', sample=None, sub_dir=None):
    '''Recursively finds all files matching the pattern.'''
    # subdir sould be a string, for example "abc03",
    # meaning we take data from directories a,b,c,0 and 3

    files = []
    groups = []
    directories = []

    # TODO : add lines to check format of input
    # TODO : try/except ?

    if sub_dir!=None :
        for c in sub_dir :
            directories.append(directory + c + "/")
    else :
        directories.append(directory)

    for path in directories :
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))

    # add randomization here maybe

    total_nb = len(files)
    if sample != None:
        total_nb = sample
    nb_full_groups, rest = np.divmod(total_nb, group_size)

    for i in range(nb_full_groups) :
        # add randomization here of files[i:i+group_size] maybe
        groups.append(files[i * group_size : (i+1) * group_size])

    # handle last group if group_size doesn't divide the total number of files
    if rest != 0:
        groups.append(files[nb_full_groups * group_size : nb_full_groups * group_size + rest])

    return groups
