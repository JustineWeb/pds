import tensorflow as tf


def initialize_tf_model(input_shape, nb_labels, nb_batch, batch_size, learning_rate, is_training=True) :
    print("Initialize tf model ...")
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=(input_shape*nb_batch, batch_size, 1))
    y = tf.placeholder(tf.float32, shape=(input_shape*nb_batch, nb_labels))

    net = build_model(x, is_training=is_training, config=BASIC_CONFIG)
    predictions = tf.layers.dense(net, nb_labels, activation=tf.sigmoid)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = predictions)
    reduced_loss = tf.reduce_mean(loss)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(reduced_loss)
    auc = tf.metrics.auc(labels = y, predictions=predictions)

    # Saver for storing checkpoints of the model. (Wavenets)
    saver = tf.train.Saver(var_list=tf.global_variables())

    return x, y, net, predictions, loss, reduced_loss,  train_op, auc, saver


def train(data_dir, group_size, sample_size, file_type, train_dir, epochs, \
          labels, labels_name, nb_labels, batch_size, nb_batch, logdir, learning_rate) :

    pattern = ""
    if file_type == "mp3" :
        pattern = "*.mp3"
    else :
        if file_type == "wav" :
            pattern = "*.wav"
        else :
            print("Argument should be either \"mp3\" or \"wav\".")

    # keep results for plotting
    train_loss_results = []
    train_auc_results = []


    files_by_group = find_files_group_select(data_dir, labels, labels_name, group_size, sample=sample_size, \
                                             pattern=pattern, sub_dir=train_dir)

    n_groups = len(files_by_group)
    #print(data_dir, labels, labels_name, group_size)

    x, y, net, predictions, loss, reduced_loss, \
    train_op, auc, saver = initialize_tf_model(len(files_by_group[0]), nb_labels, nb_batch,\
                                               batch_size, learning_rate)
    print("Start training...")

    start = time.time()

    with tf.Session() as sess:

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init)

        # Go through the whole DS at each EPOCH
        for epoch in tqdm(range(epochs)):

            t0_epoch = time.time()

            # Group by group
            for count, g in enumerate(files_by_group) :

                # Load audio and labels
                tload0 = time.time()
                audios, tags = load_audio_label_aux(labels, g, len(data_dir), labels_name=labels_name, \
                        nb_labels=nb_labels, file_type=file_type, batch_size=batch_size, nb_batch=nb_batch)

                tload1 = time.time()

                if count==0 :
                    print(">> Total loading time : {:.2f} sec".format(tload1-tload0))
                #audio_tf = tf.convert_to_tensor(audios, np.float32)

                # add check to verify if there is something to restore
                #saver.restore(sess, LOGDIR)

                predict, _, loss_value, auc_score = sess.run([predictions, train_op, reduced_loss, auc],\
                                                             feed_dict={x: audios, y: tags})

                auc_result, update_op = auc_score

                saver.save(sess, logdir)


                train_loss_results.append(loss_value)
                train_auc_results.append(auc_result)

                if (count % 20) == 0 :
                    print("Group {} done. {} left.".format(count, n_groups-count-1))


            t1_epoch = time.time()

            dur = t1_epoch-t0_epoch

            print("Iter: {:3}, Time (in sec): {:.2f}, Loss: {:.4f}, AUC : {:.4f}"\
                  .format(epoch, dur, loss_value, auc_result))
            print()

    end = time.time()
    duration2 = end-start
    print("Total time: {:.2f} sec.".format(duration2))
    return train_loss_results, train_auc_results
