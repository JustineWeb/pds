import numpy as np
from matplotlib import pyplot as plt

# ------------------------------------- #
# CODE USED FOR THE CO-OCCURRENCE matrix
# from : https://stackoverflow.com/questions/20574257/constructing-a-co-occurrence-matrix-in-python-pandas

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: http://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    #from itertools import izip
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: http://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels):
    '''
    Inspired by:
    - http://stackoverflow.com/a/16124677/395857
    - http://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(40, 20))



def check_overlaps(label_headers, label_data):


    #print('labels:\n{0}'.format(label_data))

    # Compute cooccurrence matrix
    cooccurrence_matrix = np.dot(label_data.transpose(),label_data)
    #print('\ncooccurrence_matrix:\n{0}'.format(cooccurrence_matrix))

    # Compute cooccurrence matrix in percentage
    # FYI: http://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    #      http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero/32106804#32106804
    cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurrence_matrix, \
                                                                      cooccurrence_matrix_diagonal[:, None]))
    #print('\ncooccurrence_matrix_percentage:\n{0}'.format(cooccurrence_matrix_percentage))

    # Add count in labels
    label_header_with_count = [ '{0} ({1})'.format(label_header, cooccurrence_matrix_diagonal[label_number]) \
                               for label_number, label_header in enumerate(label_headers)]
    print('\nlabel_header_with_count: {0}'.format(label_header_with_count))

    # Plotting
    x_axis_size = cooccurrence_matrix_percentage.shape[0]
    y_axis_size = cooccurrence_matrix_percentage.shape[1]
    title = "Co-occurrence matrix\n"
    xlabel= ''#"Labels"
    ylabel= ''#"Labels"
    xticklabels = label_header_with_count
    yticklabels = label_header_with_count
    heatmap(cooccurrence_matrix_percentage, title, xlabel, ylabel, xticklabels, yticklabels)
    plt.savefig('image_output.png', dpi=300, format='png', bbox_inches='tight')
    # use format='svg' or 'pdf' for vectorial pictures
    #plt.show()

# ---------------------------------------------#

def label_stats(labels, header, plot=False) :
    nb_labels_per_song = labels.iloc[:,1:-1].astype(int).sum(axis=1)

    nb_song_per_label = labels.iloc[:,1:-1].astype(int).sum(axis=0)
    nb_song_per_label = nb_song_per_label.sort_values(ascending=False)
    label_header = np.asarray(list(header.values()))[1:-1]
    label_header_by_freq = np.asarray(nb_song_per_label.index)

    # plotting
    if plot :
        fig = plt.figure(figsize=(16,6))

        plot_nb = 70

        y_pos = np.arange(plot_nb)
        plt.bar(y_pos, nb_song_per_label[:plot_nb], align='center', alpha=0.5)
        plt.xticks(y_pos, nb_song_per_label[:plot_nb])
        plt.ylabel('Occurence')
        plt.title('Label histogram')
        plt.xticks(np.arange(plot_nb), label_header_by_freq[:plot_nb], rotation=90, fontsize = 13)

        plt.show()

    print("Number of songs : " , labels.shape[0])
    print("Number of labels : " , labels.shape[1]-2) # -2 is for index columns and mp3 path column
    print("Max number of songs tagged with the same label : ",max(nb_song_per_label))
    print("Max number of labels for a single song : ",max(nb_labels_per_song))

    return nb_labels_per_song, nb_song_per_label, label_header_by_freq
