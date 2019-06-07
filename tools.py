from matplotlib import pyplot as plt

def return_params_mp3_wav(string, DATA_DIRECTORY, WAV_DIRECTORY) :
    # return : directory of files, pattern for find_files_function, file_type
    if string == "mp3" :
        return DATA_DIRECTORY, '*.mp3', "mp3"
    if string == "wav" :
        return WAV_DIRECTORY, '*.wav', "wav"
    else :
        print("Argument should be either \"mp3\" or \"wav\".")
        return

def plot_auc_loss(train_loss_results, train_auc_results):
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("AUC score", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_auc_results)
    plt.show()

# Useful fonctions for preprocessing output of train functions
# before calling plot_auc_loss
def flatten_loss_auc_results(train_loss_results, train_auc_results):
    return [e for l in train_loss_results for e in l], [e for l in train_auc_results for e in l]

def average_loss_auc_results(train_loss_results, train_auc_results):
    return [sum(l) / len(l) for l in train_loss_results], [sum(l) / len(l) for l in train_auc_results]
