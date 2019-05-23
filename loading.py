import pandas as pd
import numpy as np
import random
import os
import warnings
import fnmatch
import time
import librosa

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

# Function to lead the labels file (csv)
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
def load_audio_label_aux(labels, filenames, prefix_len, labels_name, nb_labels, \
                         file_type, batch_size, nb_batch):

    assert (file_type=="wav" or file_type=="mp3"), "The argument file_type should be either 'wav', either 'mp3'."

    nb_songs = len(filenames)

    if nb_songs > 20 :
        warnings.warn("The argument num_song should not be too high (above 20), make sure this will \
        not cause memory error.", FutureWarning, stacklevel=2)


    print("Loading {} songs ...".format(nb_songs))

    start = time.time()

    audios = np.ndarray(shape=(nb_songs * nb_batch, batch_size, 1), dtype=np.float32, order='F')
    tags = np.ndarray(shape=(nb_songs * nb_batch, nb_labels), dtype=np.float32, order='F')

    #count = 0

    idx = 0

    for f in filenames:

        # Load audio (MP3/WAV) file
        try :
            audio, _ = librosa.load(f, sr=None, mono=True)
        except EOFError :
            print("EOFERROR : The following file could not be loaded with librosa - ", f)

        audio = audio.reshape(-1, 1)

        for n in range(nb_batch) :
            audios[idx] = audio[n*batch_size: (n+1)*batch_size,:]

            # take labels or corresponding song

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

# selective version of above function
def load_audio_label_aux_selective(labels, filenames, prefix_len, labels_name, nb_labels, \
                         file_type, batch_size, nb_batch):

    assert (file_type=="wav" or file_type=="mp3"), "The argument file_type should be either 'wav', either 'mp3'."

    if nb_labels < 2 :
        print("The function load_audio_label_aux_selective should only be called when picking \
        at least 2 labels.")
        return

    nb_songs = len(filenames)

    if nb_songs > 20 :
        warnings.warn("The argument num_song should not be too high (above 20), make sure this will \
        not cause memory error.", FutureWarning, stacklevel=2)


    print("Loading {} songs ...".format(nb_songs))

    start = time.time()

    audios = np.ndarray(shape=(nb_songs * nb_batch, batch_size, 1), dtype=np.float32, order='F')
    tags = np.ndarray(shape=(nb_songs * nb_batch, nb_labels), dtype=np.float32, order='F')

    #count = 0
    select_labels = labels[labels_name]
    rows = pd.concat((select_labels[lab] == 1 for lab in labels_name), axis=1).any(axis=1)
    print(type(rows))
    print("OKKKKKKK")
    #select_labels = select_labels[rows]

    filenames_select = labels.loc[rows]['mp3_path']
    print(len(filenames_select))

    idx = 0

    for f in filenames:

        # Load audio (MP3/WAV) file
        try :
            audio, _ = librosa.load(f, sr=None, mono=True)
        except EOFError :
            print("EOFERROR : The following file could not be loaded with librosa - ", f)

        audio = audio.reshape(-1, 1)

        for n in range(nb_batch) :

            # take labels of corresponding song

            if file_type=="mp3" :
                select_labels  = labels.loc[labels['mp3_path']==f[prefix_len:]]

            if file_type=="wav" :
                select_labels  = labels.loc[labels['mp3_path']==f[prefix_len:-4]+".mp3"]

            # select wanted labels
            # >> selective version : only songs with at least one label

            # other verison to try ?
            # select_labels = select_labels.loc[(select_labels[labels_name] == 1).any(axis=1)]
            audios[idx] = audio[n*batch_size: (n+1)*batch_size,:]
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

    return audios[rows], tags
