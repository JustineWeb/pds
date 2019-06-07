import librosa
import os
import time

# clean this later ?
# >>> version on my local machine
MTT_DIR = "../MTT/"
DATA_DIRECTORY = MTT_DIR + "dataset/"
WAV_DIRECTORY = MTT_DIR + "wav-dataset/"

# >>> version on the LTS2 server
#MTT_DIR = "/mnt/scratch/students/jjgweber-MagnaTagATune/"
#DATA_DIRECTORY = MTT_DIR + "dataset/"
#WAV_DIRECTORY = MTT_DIR + "wav-dataset/"

# >>> usage in jupyter notebook
#from mp3_to_wav import *
#files = find_files(DATA_DIRECTORY, sample=3)
#from_mp3_to_wav(files, WAV_DIRECTORY)

def from_mp3_to_wav(files, wav_dir):
    print("Start converting files from mp3 to wav...")
    t1a = time.time()
    count = 0
    for file_name in files :

        file_name_cut = file_name[len(DATA_DIRECTORY):-4]

        dir_path = wav_dir + file_name_cut[:2]

        new_file_name = wav_dir + file_name_cut + ".wav"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        try :
            y, sr = librosa.load(file_name, sr=None, mono=True)
            librosa.output.write_wav(new_file_name, y, sr)

        except EOFError :
            print("EOFERROR : The following file could not be loaded with librosa - ", file_name)

        count +=1
        if (count % 200) == 0:
            print(count)

    t2a = time.time()
    print("Converted {} files.".format(count))
    print("From mp3 to wav time : {:.2f} hours".format((t2a-t1a)/3600))


def main() :
    t0a = time.time()
    files = find_files(DATA_DIRECTORY)
    t1a = time.time()
    from_mp3_to_wav(files, WAV_DIRECTORY)
    t2a = time.time()

    print("Find files time : {:.3f} sec".format(t1a-t0a))

    print("Time to convert 1 song from mp3 to wav : {:.4f} sec".format((t2a-t1a)/FILE_NB))

    return None

if __name__ == '__main__':
    main()
