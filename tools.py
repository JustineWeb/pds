def return_params_mp3_wav(string, DATA_DIRECTORY, WAV_DIRECTORY) :
    # return : directory of files, pattern for find_files_function, file_type
    if string == "mp3" :
        return DATA_DIRECTORY, '*.mp3', "mp3"
    if string == "wav" :
        return WAV_DIRECTORY, '*.wav', "wav"
    else :
        print("Argument should be either \"mp3\" or \"wav\".")
        return
