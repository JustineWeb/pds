import tensorflow as tf
from tools import *
from ckpt_saving import *
from loading import *
from models import *
from tqdm import tqdm_notebook as tqdm


def initialize_tf_model(config, is_training=True) :

    input_size = config['tr_batch_size']
    if not is_training :
        input_size = config['te_batch_size']

    nb_chunks = config['nb_chunks']
    chunk_size = config['chunk_size']

    print("Initialize tf model ...")
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=(input_size*nb_chunks, chunk_size, 1))
    y = tf.placeholder(tf.float32, shape=(input_size*nb_chunks, config['nb_labels']))

    net = build_model(x, is_training=is_training, config=config)
    predictions = tf.layers.dense(net, config['nb_labels'], activation=tf.sigmoid)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = predictions)
    reduced_loss = tf.reduce_mean(loss)
    train_op = tf.train.AdamOptimizer(learning_rate=config['learning_rate']).minimize(reduced_loss)
    auc = tf.metrics.auc(labels = y, predictions=predictions)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=config['max_checkpoints'])

    return x, y, net, predictions, loss, reduced_loss, train_op, auc, saver



def train_and_load_restore(config, directories, labels, restore=False, restore_from=None) :
    '''Function for loading and training data (naive implementation :
    alternates between loading and training at each iteration).

    - 'config': dictionary containing all characteristics of input shape and model.
    - 'directories': dictionary containing all needed paths.
    - 'labels': pandas dataframe of all labels, containing a column for name of mp3 file.
    - 'restore': boolean, set to true if training is the continuity of a previous training.
    - 'restore_from': path of the checkpoint to restore if training is the continuity of a
                                                                        previous training.

    Usage example :
    - train_restore(BASIC_CONFIG, DIRECTORIES, labels)
    - train_restore(BASIC_CONFIG, DIRECTORIES, labels, True,'train/2019-06-04T19-12-28')
    '''

    # EXTRACT CONFIG VARIABLES
    data_dir, pattern, file_type = return_params_mp3_wav(config['file_type'], \
                                                         directories['mp3_dir'], directories['wav_dir'])

    logdir_root = directories['logdir']
    batch_size = config['tr_batch_size']
    sample_size = config['tr_sample_size']
    train_dir = config['train_dir']
    epochs = config['epochs']
    labels_name = config['labels_name']
    nb_labels = config['nb_labels']
    chunk_size = config['chunk_size']
    nb_chunks = config['nb_chunks']
    learning_rate = config['learning_rate']

    # PARAMETER CHECK
    if restore and restore_from is None :
        raise ValueError("You need to specify the checkpoint to restore from"
                         "if restore is True.")

    # CREATE CHECKPOINT DIRECTORY
    logdir = get_default_logdir(logdir_root)
    print('Using default logdir: {}'.format(logdir))

    # LOSS AND AUC RESULTS
    train_loss_results = []
    train_auc_results = []

    # GET NAME OF FILES FOR SONGS TO LOAD
    # use find_files_batch_select which filters songs
    # which have at least one of the labels and create batches
    files_by_batch = find_files_batch_select(data_dir, labels, labels_name, batch_size, sample=sample_size, \
                                             pattern=pattern, sub_dir=train_dir)

    n_batches = len(files_by_batch)

    # INITIALIZE TF MODEL (VARIABLES)
    x, y, net, predictions, loss, reduced_loss, train_op, auc, saver = initialize_tf_model(config)

    # TRAINING
    print("Start training...")

    start = time.time()

    with tf.Session() as sess:

        # INITIALIZE SESSION AND ADAPT IF RESTORING
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init)

        last_saved_step = -1
        # The first training step will be saved_global_step + 1,
        # therefore we put -1 here for new or overwritten trainings.
        saved_global_step = -1

        if restore :
            try:
                # compute number of epochs left
                saved_global_step = load(saver, sess, directories['logdir']+restore_from)
                last_saved_step = saved_global_step

            except:
                print("Something went wrong while restoring checkpoint. "
                      "We will terminate training to avoid accidentally overwriting "
                      "the previous model.")
                raise


        # START TRAINING
        epoch = 0
        idx_loss_auc = 0
        try:
            for epoch in tqdm(range(saved_global_step + 1,  config['epochs'])):

                t0_epoch = time.time()
                loss_values = []
                auc_results = []

                # Feed songs to the network batch by batch
                for count, g in enumerate(files_by_batch) :

                    # Load audio and labels
                    tload0 = time.time()
                    audios, tags = load_audio_label_aux(labels, g, len(data_dir), config)

                    tload1 = time.time()

                    if count==0 :
                        print(">> Total loading time : {:.2f} sec".format(tload1-tload0))

                    # Feed to network and get metrics
                    predict, _, loss_value, auc_score = sess.run([predictions, train_op, reduced_loss, auc],\
                                                                 feed_dict={x: audios, y: tags})

                    auc_result, update_op = auc_score

                    # Save loss and auc values
                    loss_values.append(loss_value)
                    auc_results.append(auc_result)

                    if (count % 20) == 0 :
                        print("Group {} done. {} left.".format(count, n_batches-count-1))


                train_loss_results.append(loss_values)
                train_auc_results.append(auc_results)

                dur = time.time()-t0_epoch

                try :
                    mean_loss = sum(train_loss_results[idx_loss_auc]) / len(train_loss_results[idx_loss_auc])
                    mean_auc = sum(train_auc_results[idx_loss_auc]) / len(train_auc_results[idx_loss_auc])
                except :
                    print(idx_loss_aux)
                    print(train_loss_results)
                    print(train_auc_results)

                idx_loss_auc +=1

                print("Epoch: {:3}, Time (in sec): {:.2f}, Loss: {:.4f}, AUC : {:.4f}"\
                      .format(epoch, dur, mean_loss, mean_auc))
                print()

                # Save network state periodically
                if epoch % config['checkpoint_every'] == 0:
                    save(saver, sess, logdir, epoch)
                    last_saved_step = epoch

        except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
            print()

        finally:
            if epoch > last_saved_step:
                save(saver, sess, logdir, epoch)

            duration2 = time.time()-start
            print("Total time: {:.2f} minutes.".format(duration2/60))

    return train_loss_results, train_auc_results, logdir



def testing_validing(config, directories, labels, model_ckpt, validation=False) :
    '''Function for testing model or for validation.

    - 'config': dictionary containing all characteristics of input shape and model.
    - 'directories': dictionary containing all needed paths.
    - 'labels': pandas dataframe of all labels, containing a column for name of mp3 file.
    - 'model_ckpt': path of the checkpoint to restore to test / validate it
    - 'validation': boolean, if set to true predictions will be done on the training set.

    Usage example :
    - testing_validing(BASIC_CONFIG, DIRECTORIES, labels, 'train/2019-06-04T19-12-28', validation=False)
    '''

    # EXTRACT CONFIG VARIABLES
    data_dir, pattern, file_type = return_params_mp3_wav(config['file_type'], \
                                                         directories['mp3_dir'], directories['wav_dir'])

    logdir_root = directories['logdir']

    batch_size = config['te_batch_size']
    sample_size = config['te_sample_size']
    test_dir = config['test_dir']

    # validation means we test on the training set
    # in that case we overwrite the variable above
    # we consider the size of the validation set
    # is the same as the test set.
    if validation :
        test_dir = config['train_dir']

    epochs = config['epochs']
    labels_name = config['labels_name']
    nb_labels = config['nb_labels']
    chunk_size = config['chunk_size']
    nb_chunks = config['nb_chunks']
    learning_rate = config['learning_rate']

    # LOSS AND AUC RESULTS
    test_loss_results = []
    test_auc_results = []

    # GET NAME OF FILES FOR SONGS TO LOAD
    # use find_files_batch_select which filters songs
    # which have at least one of the labels and create batches
    files_by_batch = find_files_batch_select(data_dir, labels, labels_name, batch_size, sample=sample_size, \
                                             pattern=pattern, sub_dir=test_dir)

    n_batches = len(files_by_batch)

    # INITIALIZE TF MODEL (VARIABLES)
    x, y, net, predictions, loss, reduced_loss, train_op, auc, _ = initialize_tf_model(config, is_training=False)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(variables_to_restore)

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        print('Restoring model from {}'.format(model_ckpt))
        saver.restore(sess,  directories['logdir']+model_ckpt)
        print("Model restored.")

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init)

        # START TESTING EACH GROUP
        try:
            t0_epoch = time.time()

            # Feed songs to the network batch by batch
            for count, g in enumerate(files_by_batch) :

                # Load audio and labels
                tload0 = time.time()
                audios, tags = load_audio_label_aux(labels, g, len(data_dir), config)

                tload1 = time.time()
                print(">> Total loading time : {:.2f} sec".format(tload1-tload0))

                # Feed to network and get metrics
                predict, _, loss_value, auc_score = sess.run([predictions, train_op, reduced_loss, auc],\
                                                             feed_dict={x: audios, y: tags})

                auc_result, update_op = auc_score

                # Save loss and auc values
                test_loss_results.append(loss_value)
                test_auc_results.append(auc_result)

                dur = time.time()-t0_epoch

                print("Group {} done. {} left.".format(count, n_batches-count-1))
                print("Time (in sec): {:.2f}, Loss: {:.4f}, AUC : {:.4f}"\
                  .format(dur, loss_value, auc_result))

                print()


        except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
            print()

        finally:
            duration2 = time.time()-t0_epoch
            print("Total time: {:.2f} minutes.".format(duration2/60))

    return test_loss_results, test_auc_results
