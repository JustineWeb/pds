import pandas as pd
import numpy as np
import random
import os
import warnings
import fnmatch
import time
import librosa

# not used
FILE_PLACEHOLDER = "placeholder"

def load_labels(labels_file_name):
    pd.read_csv(labels_file_name)

def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


# original version not used anymore
def find_files(directory, pattern='*.mp3', sample=None, sub_dir=None):
    '''Recursively finds all files matching the pattern.'''
    # subdir sould be a string, for example "abc03",
    # meaning we take data from directories a,b,c,0 and 3

    files = []
    directories = []

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

# not used anymore
def find_files_select(directory, labels, labels_name, pattern='*.mp3', sample=None, sub_dir=None):
    '''Recursively finds all files matching the pattern,
    Which are tagged with at least one of the labels in
    the list labels, given as parameter.'''
    # subdir sould be a string, for example "abc03",
    # meaning we take data from directories a,b,c,0 and 3
    _, select_filenames = sublabels(labels_name, labels)

    # remove the c/ at the beginning of each filename for comparison later
    select_filenames = [s[2:] for s in select_filenames.values]
    files = []
    directories = []

    if sub_dir!=None :
        for c in sub_dir :
            directories.append(directory + c + "/")
    else :
        directories.append(directory)

    for path in directories :
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, pattern):
                if filename in select_filenames :
                    files.append(os.path.join(root, filename))

    if sample!=None :
        try:
            return files[:sample]
        except TypeError:
            print("Argument sample should be either None, or an integer :\
             the number of first n samples to take.")
    else :
        return files


# not used anymore
def find_files_group(directory, group_size, pattern='*.mp3', sample=None, sub_dir=None):
    '''Recursively finds all files matching the pattern
    And split them into groups of size groupe_size.
    Useful for feeding group by group to th network.'''
    # subdir sould be a string, for example "abc03",
    # meaning we take data from directories a,b,c,0 and 3

    files = []
    groups = []
    directories = []

    if sub_dir!=None :
        for c in sub_dir :
            directories.append(directory + c + "/")
    else :
        directories.append(directory)

    for path in directories :
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))

    # Randomization should be added here

    total_nb = len(files)
    if sample != None:
        total_nb = sample
    nb_full_groups, rest = np.divmod(total_nb, group_size)

    for i in range(nb_full_groups) :
        # Randomization should be added here of files[i:i+group_size]
        groups.append(files[i * group_size : (i+1) * group_size])

    # handle last group if group_size doesn't divide the total number of files
    # For now this never happens as the sample size is always made a multiple
    # of the batch size. If one wants to adapt it, it should modify some part
    # of function load_audio_label_aux.
    if rest != 0:
        groups.append(files[nb_full_groups * group_size : nb_full_groups * group_size + rest])
        # "zero-padding" to make the last group in the shape expected by the nn
        # > here we add a None element and when loading we check whether the files
        # name is None and add zeros in that case
        groups[nb_full_groups] += [FILE_PLACEHOLDER] * (group_size - rest)

    return groups

# This is the version currently used: a combination of all functionalities
# provided in the above functions.
def find_files_group_select(directory, labels, labels_name, group_size, pattern='*.mp3', sample=None, sub_dir=None):
    '''Recursively finds all files matching the pattern,
    which are tagged with at least one of the labels in
    the list labels, given as parameter
    and split them into groups of size groupe_size.
    Useful for feeding group by group to th network.'''

    # subdir sould be a string, for example "abc03",
    # meaning we take data from directories a,b,c,0 and 3 >> useful if we want to separate between
    # train and test sets for example

    # extract all filenames from labels file for which there is at least
    # one of the given labels.
    _, select_filenames = sublabels(labels_name, labels)

    # remove the c/ at the beginning of each filename for comparison
    # with label columns name later
    select_filenames = [s[2:] for s in select_filenames.values]

    files = []
    groups = []
    directories = []

    # iterate over all given subdirectories
    if sub_dir!=None :
        for c in sub_dir :
            directories.append(directory + c + "/")
    else :
        directories.append(directory)

    for path in directories :
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, pattern):
                # filter to take songs which only have at least one label
                if filename in select_filenames :
                    files.append(os.path.join(root, filename))

    # Randomization should be added here

    total_nb = len(files)

    if sample > total_nb :
        warnings.warn("The argument sample should be smaller than the number of available songs ({}).\
        Otherwise it will simply be ignored. Make sure this is not an error.".format(total_nb), FutureWarning, stacklevel=2)

    if sample != None and sample < total_nb :
        total_nb = sample
    nb_full_groups, rest = np.divmod(total_nb, group_size)

    for i in range(nb_full_groups) :
        # Randomization should be added here of files[i:i+group_size]
        groups.append(files[i * group_size : (i+1) * group_size])

    # handle last group if group_size doesn't divide the total number of files
    # For now this never happens as the sample size is always made a multiple
    # of the batch size. If one wants to adapt it, it should modify some part
    # of function load_audio_label_aux.
    if rest != 0:
        groups.append(files[nb_full_groups * group_size : nb_full_groups * group_size + rest])

        # "zero-padding" to make the last group in the shape expected by the nn
        # > here we add a None element and when loading we check whether the files
        # name is None and add zeros in that case
        groups[nb_full_groups] += [FILE_PLACEHOLDER] * (group_size - rest)

    return groups


def sublabels(labels_name, labels):

    labels_name_ext = [n for n in labels_name]
    labels_name_ext.append('mp3_path')
    select_labels = labels[labels_name_ext]

    rows = pd.concat((select_labels[lab] == '1' for lab in labels_name), axis=1).any(axis=1)

    select_labels = select_labels[rows]

    select_filenames = select_labels['mp3_path']
    print("All labels : {} songs >>> Selected for given labels : {}. (test or train sets are note taken\
    into account here)".format(len(labels),len(select_labels)))

    return select_labels, select_filenames


# Function to load the labels file (csv)
def load_and_clean_labels(labels_path):
    start = time.time()
    labels = pd.read_csv(labels_path, sep = '"\t"')
    end = time.time()
    print("Loading csv file : {:.3}".format(end-start))

    # Prepare header to put back in the end
    # remove quotes and take all columns except the first one
    header = list(map(lambda x : x.replace('"', ''), labels))[1:]
    # add back the first column, separated in two
    header = ['clip_id', 'no_voice']+header
    # create dictionary
    header = dict(enumerate(header))

    # Solve format problem : two first columns are merged
    # extract first column and rest
    left, right = labels['"clip_id\t""no voice"'], labels.iloc[:, 1:]
    # split first column in two part at separator "\t"
    split = left.str.split(pat = "\t", expand=True).replace('"', '')

    # put back the first column which is now two, with the rest
    cleaned = pd.concat([split, right], axis=1, ignore_index=True)
    # clean by removing quotes and add back header
    cleaned = cleaned.apply(lambda col : col.apply(lambda x : x.replace('"', ''))).rename(columns = header)

    return cleaned, header

# Find files is done externally, here we give the file names as parameters
def load_audio_label_aux(labels, filenames, prefix_len, config):
    '''Function for loading audios and labels.
    "aux" stands for auxiliary because originally finding the filenames
    was also done in the loading function.

    - 'labels': pandas dataframe of all labels, containing a column for name of mp3 file.
    - 'filenames': python list containing all filenames we wish to load
    - 'prefix_len': length of part of the path we don't need (typically 12 for len(MTT/dataset/))
    - 'config': dictionary containing all characteristics of input shape and model.

    Usage example :
    - load_audio_label_aux(labels, files_list, len(data_dir), BASIC_CONFIG)
    '''

    labels_name = config['labels_name']
    nb_labels = config['nb_labels']
    file_type = config['file_type']
    chunk_size = config['chunk_size']
    nb_chunks = config['nb_chunks']

    nb_songs = len(filenames)

    if nb_songs > 20 :
        warnings.warn("The argument num_song should not be too high (above 20), make sure this will \
        not cause memory error.", FutureWarning, stacklevel=2)

    #print("Loading {} songs ...".format(nb_songs))

    start = time.time()

    audios = np.ndarray(shape=(nb_songs * nb_chunks, chunk_size, 1), dtype=np.float32, order='F')
    tags = np.ndarray(shape=(nb_songs * nb_chunks, nb_labels), dtype=np.float32, order='F')

    #count = 0

    idx = 0

    for f in filenames:

        # For now this never happens as we always make the sample size be a multiple of the batch_size
        # If one wants to adapt this it should modify it because it doesn't work the it is written here
        if f == FILE_PLACEHOLDER :
            for n in range(nb_chunks) :
                audios[idx] = [[0]] * chunk_size

                # the values are very weird here eg [1.92838320e+31 2.35106045e-38]
                tags[idx] = [0.0] * nb_labels

        else :
            # Load audio (MP3/WAV) file
            try :
                audio, _ = librosa.load(f, sr=None, mono=True)
            except EOFError :
                print("EOFERROR : The following file could not be loaded with librosa - ", f)

            audio = audio.reshape(-1, 1)

            for n in range(nb_chunks) :
                audios[idx] = audio[n*chunk_size: (n+1)*chunk_size,:]

                # take labels of corresponding song

                if file_type=="mp3" :
                    select_labels  = labels.loc[labels['mp3_path']==f[prefix_len:]]

                if file_type=="wav" :
                    select_labels  = labels.loc[labels['mp3_path']==f[prefix_len:-4]+".mp3"]

                # select wanted labels
                select_labels = select_labels[labels_name]

                tags[idx] = select_labels.values.reshape(nb_labels)

                idx +=1

        #count +=1
        #if (count % 10) == 0:
         #   print(count)

    end = time.time()
    duration = end-start

    #print(">> Total loading time - {} songs : {:.2f} sec".format(nb_songs, duration))
    #print()
    #print("Shape of audios list :", audios.shape)
    #print("Shape of tags list :", tags.shape)
    #print()
    return audios, tags
