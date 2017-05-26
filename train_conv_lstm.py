import tensorflow as tf
from tensorflow.python import debug as tf_debug
import datumio.datagen as dtd
from tqdm import *
import fire
import h5py
import numpy as np
import os
from conv_lstm_model import conv_lstm


def regression_loss(reg_preds, reg_labels):
    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(reg_labels, reg_preds)))
    tf.add_to_collection('losses', rmse)
    return rmse, tf.add_n(tf.get_collection("losses"), name="total_loss")


def combind_loss(logits, labels, reg_preds, reg_labels):
    alpha = 1
    beta = 0.025
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cem = tf.reduce_mean(cross_entropy, name='cross_entropy')
    w_cem = cem * alpha
    tf.add_to_collection("losses", w_cem)
    reg_labels = tf.reshape(reg_labels, (-1, 1))
    # rmse = tf.sqrt(tf.losses.mean_squared_error(reg_labels, reg_preds, loss_collection=None))
    rmse = regression_loss(reg_preds, reg_labels)
    w_rmse = rmse * beta
    tf.add_to_collection("losses", w_rmse)

    return tf.add_n(tf.get_collection("losses"), name='combinded_loss'), cem, rmse

def h5reader(fname, set_type):
    if set_type == "train":
        train_x_name = set_type + "_set_x"
        train_y_name = set_type + "_set_y"
        train_c_name = set_type + "_set_class"
        with h5py.File(fname, "r") as f:
            train_set_x = f[train_x_name][:]
            train_set_y = f[train_y_name][:]
            train_set_c = f[train_c_name][:]

        return train_set_x, train_set_y, train_set_c
    else:
        test_x_name = set_type + "_set_x"
        with h5py.File(fname, "r") as f:
            test_set_x = f[test_x_name][:]

        return test_set_x

def train():
    # ===============configuration parameters=============
    batch_size = 32
    time_step = 15
    display_step = 10
    n_epochs = 1000
    snap_step = 616
    start_lr = 0.0001
    lr_decay_step = 2000
    lr_decay_rate = 0.95
    n_class = 22
    n_batch = 313
    dtype = tf.float32
    debug = False
    output = True
    finetuning = False

    dirpath = "../result/conv_lstm_bs_32_channel_3_resampled_5_conv_2_fc"
    train_fname = "../data/CIKM2017_train/train_Imp_3x3_del_height_no.4_classified.h5"
    test_fname = "../data/CIKM2017_testA/testA_Imp_3x3_del_height_no.4.h5"
    checkpoint_dir = os.path.join(dirpath, "ckeckpoints")
    log_dir = os.path.join(dirpath, "logs")
    output_dir = os.path.join(dirpath,"outputs")

    # get data
    train_set_x, train_set_y, train_set_c = h5reader(fname=train_fname, set_type="train")
    test_set_x = h5reader(fname=test_fname, set_type="testa")

    # some placeholders
    is_training = tf.placeholder_with_default(True, None)
    keep_prob = tf.placeholder(dtype=dtype, shape=None)
    x = tf.placeholder(dtype=dtype, shape=[None, time_step, 80, 80, 3])
    y_ = tf.placeholder(dtype=dtype, shape=[None, 1])

    # some tf training parameters
    with tf.variable_scope('step'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
    tf.summary.scalar('global_step', global_step)

    lr = tf.train.exponential_decay(start_lr, global_step, lr_decay_step, lr_decay_rate, staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # model
    preds, _ = conv_lstm(x, time_step, keep_prob, dtype)

    # Define loss and optimizer
    r_loss, total_loss = regression_loss(preds, y_)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)
    rmse = tf.reduce_mean(r_loss)
    # define saver
    saver = tf.train.Saver(var_list=tf.trainable_variables() + [global_step], max_to_keep=15)

    summary_op = tf.summary.merge_all()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # create batch generator with random augmentations applied on-the-fly
    train_rng_aug_params = {'rotation_range': (-20, 20),
                      'translation_range': (-4, 4),
                      'do_flip_lr': True,
                      'do_flip_ud':True,
                      'output_shape':(80, 80)}

    test_rng_aug_params = {'output_shape':(80, 80)}

    # train set data generator
    datagen = dtd.BatchGenerator(X=train_set_x, y=train_set_y, rng_aug_params=train_rng_aug_params)

    # resample the dataset
    datagen.resample_dataset(train_set_c, "balanced")

    testgen = dtd.BatchGenerator(X=test_set_x, rng_aug_params=test_rng_aug_params)

    testgen.mean = datagen.mean
    testgen.std = datagen.std

    # Session Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.device('/gpu:0'), tf.Session(config=config) as sess:
        if debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        sess.run(init_op)
        train_writer = tf.summary.FileWriter(logdir=os.path.join(log_dir, "train"), graph=sess.graph)

        # Load checkpoint
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("checkpoint loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old checkpoint")

        for i in range(n_epochs):
            for idx, batch in enumerate(datagen.get_batch(batch_size=batch_size,buffer_size=1024,shuffle=True)):
                step_n = idx + 1 + i * n_batch
                r_l, t_l, _, summary_str = sess.run([r_loss, total_loss, train_op, summary_op], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                train_writer.add_summary(summary_str, global_step=idx*(i+1))

                if idx % display_step == 0:
                    print("... idx %d, epoch %d/%d, rmse %g, total loss %g" %(idx, i+1, n_epochs, r_l, t_l))

                if step_n % snap_step == 0:
                    if not os.path.exists(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
                    saver.save(sess, checkpoint_dir + '/' + 'checkpoints', global_step=(idx + i*n_batch))
                    print("Model save in file %s" % checkpoint_dir)

                if step_n % lr_decay_step == 0:
                    print("Learning rate decay, current learning rate:%g" % lr.eval())

            if output:
                out = []
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                for batch in testgen.get_batch(batch_size=batch_size):
                    out.append(sess.run(preds, feed_dict={x:batch,keep_prob:1.0}))
                out = np.asarray(out, dtype=np.float16)
                print(out.shape)
                out = out.reshape(-1)
                output_fname = output_dir + "/testa_epoch_%d-%d_" %(i+1, n_epochs) + ".csv"
                np.savetxt(fname=output_fname, X=out, delimiter="")
                print("testa output in file %s" % (output_fname))

            train_rmse = rmse.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("... epoch %d/%d, training rmse %g" % (i + 1, n_epochs, train_rmse))

if __name__ == "__main__":
    train()