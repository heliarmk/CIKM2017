import tensorflow as tf
import tensorlayer as tl
import numpy as np
import joblib
import fire
import time
import os
from tqdm import *
from old_conv3d_model import conv3dnet
from c3d_model import c3d_fc, c3d_fcn
from tensorflow.python import debug as tf_debug
from sklearn.metrics import mean_squared_error


# def loss(logits, labels):
#    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
#        logits, labels, name='cross_entropy_per_example')

#    return tf.reduce_mean(cross_entropy, name='xentropy_mean')
def get_num_records(tf_record_file):
    """
    calc the number of file in tf_record_file
    :param tf_record_file: the filename of tfrecords
    :return: 
    """
    return len([x for x in tf.python_io.tf_record_iterator(tf_record_file)])

def regression_loss(reg_preds, reg_labels):
    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(reg_labels, reg_preds)))
    tf.add_to_collection('losses', rmse)
    return tf.add_n(tf.get_collection('losses'), name="total_loss")

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
    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(reg_labels, reg_preds)))
    w_rmse = rmse * beta
    tf.add_to_collection("losses", w_rmse)

    return tf.add_n(tf.get_collection("losses"), name='combinded_loss'), cem, rmse


def read_data_from_pkl(datafile):
    """
    read file in joblib.dump pkl
    :param datafile: filename of pkl
    :return: 
    """
    datas = joblib.load(datafile)
    for i in range(10):
        datas = np.random.permutation(datas)
    inputs, labels = [], []
    for data in datas:
        inputs.append(data["input"])
        labels.append(data["label"])

    inputs = np.array(inputs).reshape(-1, 15, 101, 101, 3).astype(np.float32)
    inputs -= np.mean(inputs, axis=(2, 3), keepdims=True)
    inputs /= np.std(inputs, axis=(2, 3), keepdims=True)
    labels = np.array(labels).reshape(-1, 1).astype(np.float32)

    return inputs, labels


def preprocessing(inputs):
    raise NotImplementedError


def read_and_decode_from_tfrecord(trainfilename=None, validfilename=None, testfilename=None, is_training=True,
                                  batch_size=None, shuffle=False,
                                  num_epochs=None, istesta=False):
    """
    Read trainfile and validfile to produce two queues
    :param trainfilename: training tfrecord filename 
    :type string
    :param validfilename: validation tfrecord filename
    :type string
    :param is_training: selsct the input queue
    :type is_training: Bool
    :param batch_size: batch size of out put var
    :type int
    :param shuffle: is shuffle
    :type shuffle: Bool
    :return: 
    """
    if not istesta:
        # 根据文件名生成一个队列
        train_queue = tf.train.string_input_producer([trainfilename], num_epochs=num_epochs)

        valid_queue = tf.train.string_input_producer([validfilename], num_epochs=num_epochs)

        # 挑选文件队列，实现training的过程中测试
        queue_select = tf.cond(is_training,
                               lambda: tf.constant(0),
                               lambda: tf.constant(1))
        queue = tf.QueueBase.from_list(queue_select, [train_queue, valid_queue])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)  # 返回文件名和文件

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               'categroy': tf.FixedLenFeature([], tf.int64),
                                               'label': tf.FixedLenFeature([], tf.float32)
                                           })
        img = tf.decode_raw(features['image_raw'], tf.uint8)
        depth, height, width, channel = 15, 101, 101, 4
        img = tf.reshape(img, [depth, height, width, channel])
        img = tf.cast(img, tf.float32)
        #img = preprocessing(img)

        reg_label = features['label']
        class_label = features['categroy']
    else:
        # 根据文件名生成一个队列
        test_queue = tf.train.string_input_producer([testfilename], num_epochs=num_epochs)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(test_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               'label': tf.FixedLenFeature([], tf.float32)
                                           })
        img = tf.decode_raw(features['image_raw'], tf.uint8)
        depth, height, width, channel = 15, 101, 101, 4
        img = tf.reshape(img, [depth, height, width, channel])
        img = tf.cast(img, tf.float32)
        # img = preprocessing(img)
        reg_label = features['label']
        class_label = -1
    if shuffle:
        image_batch, reg_label_batch, class_label_batch = tf.train.shuffle_batch([img, reg_label, class_label],
                                                                                 batch_size=batch_size, capacity=5000,
                                                                                 min_after_dequeue=3000,
                                                                                 num_threads=4)
    else:
        image_batch, reg_label_batch, class_label_batch = tf.train.batch([img, reg_label, class_label],
                                                                         batch_size=batch_size, capacity=5000,
                                                                         num_threads=4)

    return image_batch, class_label_batch, reg_label_batch


def ten_fold_ensamble():
    trainfile_base = "../data/CIKM2017_train/10_fold/train_Imp_3x3_resample_normalization"
    validfile_base = "../data/CIKM2017_train/10_fold/valid_Imp_3x3_resample_normalization"
    testafile_base = "../data/CIKM2017_testA/10_fold/testA_Imp_3x3_normalization"

    dirpath_base = "../result/c3d_v1.0_batch_size: 16_4_channel_resample_normalization_fc_2048_bn_10_fold"

    for i in range(4, 10):
        print("==============The No.%d Fold==============" % (i))
        trainfile = trainfile_base + "_" + str(i) + "_fold.tfrecords"
        validfile = validfile_base + "_" + str(i) + "_fold.tfrecords"
        testafile = testafile_base + "_" + str(i) + "_fold.tfrecords"
        dirpath = os.path.join(dirpath_base, "Num" + str(i))
        train(trainfile, validfile, testafile, dirpath, i)


def train(trainfile, validfile, testafile, dirpath, fold=0):
    train_batch_size = 16
    test_batch_size = 1
    #train_set_num = 9000
    resample_train_set_num = 44253
    #valid_set_num = 1000
    resample_valid_set_num = 4917
    test_set_num = 2000
    #max_epochs = 200
    max_steps = 19340
    DISPLAY_STEP = 1
    testa_out_list = []
    START_LR = 0.0001
    SNAPSTEP = 1000
    LR_DECAY_STEP = 2000
    LR_DECAY_RATE = 0.95
    n_class = 22
    network_type = "FC"
    DTYPE = tf.float32
    OUTPUT = True
    DEBUG = False
    FINETUNING = False

    # dirpath = "./" + "FC_SIZE:" + str(FC_SIZE) + "_batch_size" + str(train_batch_size) + "_data_agg" + "_10_fold_cv"
    # dirpath = "../result/" + "c3d_v1.0_batch_size:" + str(train_batch_size) + "_4_channel" + "_fc" + "_bn"
    # dirpath = "../result/" + "old_conv3d_batch_size:" + str(train_batch_size) +  "_4_channel" +"_agg"
    #dirpath = "../result/c3d_v1.0_batch_size: 16_3_channel_resample_normalization_fc_2048_bn_10_fold"
    CHECKPOINT_DIR = os.path.join(dirpath, "checkpoints")
    log_dir = os.path.join(dirpath, "log")
    # read data
    # trainfile = "../data/CIKM2017_train/train_Imp_3x3_resampled.tfrecords"
    # trainfile = "../data/CIKM2017_train/train_Imp_3x3_classed.tfrecords"
    #trainfile = "../data/CIKM2017_train/train_Imp_3x3_resample_del_height_no.4_normalization.tfrecords"
    # trainfile = "../data/CIKM2017_train/train_Imp_3x3.tfrecords"
    # validfile = "../data/CIKM2017_train/valid_Imp_3x3_resampled.tfrecords"
    # validfile = "../data/CIKM2017_train/valid_Imp_3x3_classed.tfrecords"
    #validfile = "../data/CIKM2017_train/valid_Imp_3x3_resample_del_height_no.4_normalization.tfrecords"
    # validfile = "../data/CIKM2017_train/valid_Imp_3x3.tfrecords"
    #testafile = "../data/CIKM2017_testA/testA_Imp_3x3_del_height_no.4_normalization.tfrecords"

    # train_set_num = get_num_records(trainfile)
    # valid_set_num = get_num_records(validfile)

    # global variable
    is_training = tf.placeholder_with_default(True, None)
    keepprob = tf.placeholder(dtype=DTYPE, shape=None)

    with tf.variable_scope('step'):
        global_step = tf.Variable(0, name='global_step', trainable=False)

    # data input
    train_datas, train_class_labels, train_reg_labels = read_and_decode_from_tfrecord(trainfilename=trainfile,
                                                                                      validfilename=validfile,
                                                                                      is_training=is_training,
                                                                                      batch_size=train_batch_size,
                                                                                      shuffle=True)

    test_datas, test_class_labels, test_reg_labels = read_and_decode_from_tfrecord(testfilename=testafile,
                                                                                   is_training=is_training,
                                                                                   batch_size=test_batch_size,
                                                                                   shuffle=False, istesta=True)
    # test input placeholder

    # labels = tf.placeholder(dtype=DTYPE, shape=(None, 1))

    # build model
    if network_type == "FC":
        with tf.variable_scope("model") as scope:
            train_out_sm, train_out_reg, train_r_p5, train_fc1, train_fc2, train_fc3 = c3d_fc(train_datas, keepprob,
                                                                                              train_batch_size, n_class,
                                                                                              DTYPE,
                                                                                              is_training)
            scope.reuse_variables()
            test_out_sm, test_out_reg, test_r_p5, test_fc1, test_fc2, test_fc3 = c3d_fc(test_datas, keepprob,
                                                                                        test_batch_size,
                                                                                        n_class, DTYPE, is_training)

            # preds = conv3dnet(datas, keepprob, n_class)
    elif network_type == "FCN":
        with tf.variable_scope("model") as scope:
            preds, fc7_weights, reg = c3d_fcn(train_datas, keepprob, train_batch_size, n_class, DTYPE, is_training)
    else:
        raise NotImplementedError

    # some operation
    # loss
    loss, cme, rmse = combind_loss(train_out_sm, train_class_labels, train_out_reg, train_reg_labels)

    tf.summary.scalar('total_loss', loss)

    # lr
    lr = tf.train.exponential_decay(START_LR, global_step, LR_DECAY_STEP, LR_DECAY_RATE, staircase=True)
    tf.summary.scalar('learning_rate', lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if FINETUNING:
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(lr)
            grads = train_op.compute_gradients(loss, [v for v in tf.trainable_variables() if
                                                      (v.name.startswith("model/output") or v.name.startswith(
                                                          "model/fc7") or v.name.startswith("model/fc6"))])
            train_op = train_op.apply_gradients(grads, global_step=global_step)
    else:
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    summary_op = tf.summary.merge_all()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # param saver
    if FINETUNING:
        saver_finetuning = tf.train.Saver(
            [v for v in tf.trainable_variables() if not (v.name.startswith("model/output"))])

    saver = tf.train.Saver(var_list=tf.trainable_variables() + [global_step], max_to_keep=15)

    # Session Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.device('/gpu:0'), tf.Session(config=config) as sess:
        if DEBUG:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        train_writer = tf.summary.FileWriter(logdir=os.path.join(log_dir, "train"), graph=sess.graph)

        sess.run(init_op)

        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            # if FINETUNING:
            # saver_finetuning.restore(sess, checkpoint.model_checkpoint_path)
            #    print("checkpoint loaded:", checkpoint.model_checkpoint_path)
            # else:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("checkpoint loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old checkpoint")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_loss = 0
        train_cem = 0
        train_rmse = 0
        epoch = 0
        try:
            for step in range(max_steps):
                if coord.should_stop():
                    break

                _, loss_out, cem_out, rmse_out, summary_str, label, reg_pred = sess.run(
                    [train_op, loss, cme, rmse, summary_op, train_reg_labels, train_out_reg],
                    feed_dict={is_training: True, keepprob: 0.5})
                train_loss += loss_out
                train_rmse += rmse_out
                train_cem += cem_out
                train_writer.add_summary(summary_str, global_step=step)

                every_epoch_step_num = int(resample_train_set_num / train_batch_size)
                # epoch_num = int(step / every_epoch_step_num)

                if step % DISPLAY_STEP == 0 and not step % every_epoch_step_num == 0:
                    print("Step:%05d,loss:%9f,avg_cem:%9f, avg_rmse:%9f" % (
                        step, train_loss / (step % every_epoch_step_num),
                        train_cem / (step % every_epoch_step_num),
                        train_rmse / (step % every_epoch_step_num)))
                    print("tf_rmse:%9f" % (rmse_out))
                if step % LR_DECAY_STEP == 0:
                    print("Learning Rate Decay, current learning rate:%9f" % sess.run(lr))
                if step % every_epoch_step_num == 0 and not step == 0:
                    print("===============Validation===============")
                    epoch += 1
                    train_loss = 0
                    train_rmse = 0
                    train_cem = 0
                    valid_acc = 0
                    valid_cem = 0
                    valid_rmse = 0
                    valid_step = int(resample_valid_set_num / train_batch_size)
                    for i in range(valid_step):
                        loss_val, cme_val, rmse_val = sess.run([loss, cme, rmse],
                                                               feed_dict={is_training: False, keepprob: 1.0})
                        valid_acc += loss_val
                        valid_cem += cme_val
                        valid_rmse += rmse_val
                    # val_writer.add_summary(summary_str, global_step=step)

                    result = "Epoch:%03d,Acc=%9f,Cme=%9f,Rmse=%9f" % (
                        epoch, valid_acc / valid_step, valid_cem / valid_step, valid_rmse / valid_step,)
                    print(result)
                    if OUTPUT:
                        print("===============OUTPUT===============")
                        test_step = test_set_num
                        testa_out_list = []
                        for i in range(test_step):
                            reg = sess.run([test_out_reg], feed_dict={is_training: False, keepprob: 1.0})
                            testa_out_list.append(reg)
                        testa_out_list = np.array(testa_out_list).reshape(-1)
                        testa_out_dir = os.path.join(dirpath, "test_result")
                        if not os.path.exists(testa_out_dir):
                            os.mkdir(testa_out_dir)
                        test_out_fname = testa_out_dir + "/" + (
                            "Step:%05d_Valid_Rmse:%.3f.csv" % (step, valid_rmse / valid_step))
                        np.savetxt(X=testa_out_list, fname=test_out_fname, delimiter="")
                        print("Saved File in %s" % test_out_fname)
                if step % SNAPSTEP == 0:
                    if not os.path.exists(CHECKPOINT_DIR):
                        os.mkdir(CHECKPOINT_DIR)
                    saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step=step)
                    print("Model save in file %s" % CHECKPOINT_DIR)

        except tf.errors.OutOfRangeError:
            print('Done training --epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)

        '''

        train_data = base_data[ train_set_num: ]
        train_label = base_label[ train_set_num: ]
        valid_data = base_data[ 0:train_set_num ]
        valid_label = base_label[0:train_set_num ]
        train_batch_num = int(train_set_num / train_batch_size)
        valid_batch_num = int(valid_set_num / train_batch_size)
        step = 0

        for epoch in range(max_epochs):
            
            train_loss = 0
            valid_rmse = 0

            testa_out_list.clear()
            for i in range(train_batch_num):
                batch_index_front = i * train_batch_size
                batch_index_end = (i+1) * train_batch_size
                train_data_batch = np.array(train_data[batch_index_front:batch_index_end]).reshape(-1, 15, 101, 101, 4).astype(np.float32)
                train_label_batch = np.array(train_label[batch_index_front:batch_index_end]).reshape(-1,1).astype(np.float32)
                _ , l = sess.run([train_op, loss], feed_dict = {inputs: train_data_batch, labels: train_label_batch, keepprob: 0.5})
                step += 1
                train_loss += l
                if step % display_step == 0 and not (step - epoch * train_batch_num) == 0:
                    print("Step:%05d,loss:%9f" % (step, np.sqrt(train_loss / (step - epoch * train_batch_num))))
                # summary_str = sess.run(summary_op)
                # train_writer.add_summary(summary_str, global_step=step)
            
            for i in range(valid_batch_num):
                batch_index_front = i * train_batch_size
                batch_index_end = (i+1) * train_batch_size
                valid_data_batch = np.array(valid_data[batch_index_front:batch_index_end]).reshape(-1, 15, 101, 101, 4).astype(np.float32)
                valid_label_batch = np.array(valid_label[batch_index_front:batch_index_end]).reshape(-1,1).astype(np.float32)
                r = sess.run(loss, feed_dict = {inputs: valid_data_batch, labels: valid_label_batch, keepprob :1.0})
                valid_rmse += r
                # summary_str = sess.run(summary_op)
                # val_writer.add_summary(summary_str, global_step=step)

            #output testa
            for i in range(testa_data.shape[0]):
                output_batch = testa_data[i]
                testa_out = sess.run(testa_output, feed_dict = {inputs:output_batch, keepprob:1.0})
                testa_out_list.append(testa_out)

            train_loss = np.sqrt(train_loss / train_batch_num)
            valid_rmse = np.sqrt(valid_rmse / valid_batch_num)

            result = "Epoch:%03d, loss=%9f, rmse=%9f" % (epoch, train_loss, valid_rmse)

            if epoch % display_epoch == 0:
                print(result)
            
            if store_result:
                if not os.path.isdir(dirpath):
                    os.mkdir(dirpath)
                filename = dirpath + "/Epoch:" + str(epoch) + "_loss:" + str(train_loss) +"_rmse" + str(valid_rmse) + "_time:" + time.ctime() + ".pkl"
                output_dict = {"restlt": result, "output": testa_out_list}
                joblib.dump(value=output_dict, filename=filename,compress=3)

            if step % SNAPSTEP == 0:
                if not os.path.exists(CHECKPOINT_DIR):
                    os.mkdir(CHECKPOINT_DIR)
                saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step=step)
                # print("Model save in file %s" % save_path)
        '''

    tf.reset_default_graph()

def output():
    batch_size = 1
    train_set_num = 9000
    valid_set_num = 1000
    n_class = 47
    DTYPE = tf.float32
    DEBUG = False
    FINETUNING = True
    output_testA = True

    if output_testA:
        max_steps = int(2000 / batch_size)
    else:
        max_steps = int(valid_set_num / batch_size)
    # dirpath = "./" + "FC_SIZE:" + str(FC_SIZE) + "_batch_size" + str(train_batch_size) + "_data_agg" + "_10_fold_cv"
    # dirpath = "../result/" + "c3d_v1.0_batch_size:" + str(
    #    train_batch_size) + "_4_channel" + "_resample" + "_fcn" + "_bn"
    # dirpath = "../result/" + "old_conv3d_batch_size:" + str(train_batch_size) +  "_4_channel" +"_agg"
    dirpath = "../result/c3d_v1.0_batch_size: 16_4_channel_resample_normalization_fc_bn"

    CHECKPOINT_DIR = os.path.join(dirpath, "checkpoints")
    log_dir = os.path.join(dirpath, "log")
    # read data
    trainfile = "../data/CIKM2017_train/train_Imp_3x3_classed.tfrecords"
    # trainfile = "../data/CIKM2017_train/train_Imp_3x3.tfrecords"
    # trainfile = "../data/CIKM2017_train/train_Imp_3x3_fliped&rotated.tfrecords"
    validfile = "../data/CIKM2017_train/valid_Imp_3x3_classed.tfrecords"
    # validfile = "../data/CIKM2017_train/valid_Imp_3x3_fliped&rotated.tfrecords"
    testafile = "../data/CIKM2017_testA/testA_Imp_3x3_normalization.tfrecords"

    # global variable
    is_training = tf.placeholder_with_default(True, None)
    keepprob = tf.placeholder(dtype=DTYPE, shape=None)

    # data input
    datas, class_labels, reg_labels = read_and_decode_from_tfrecord(trainfile, testafile, is_training, batch_size,
                                                                    shuffle=False, istesta=True)

    # build model
    with tf.variable_scope("model") as scope:
        sm, reg, r_p5, fc1, fc2, fc3 = c3d_fc(datas, keepprob, batch_size, n_class, DTYPE, is_training)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver(tf.all_variables())

    # Session Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.device('/gpu:0'), tf.Session(config=config) as sess:

        sess.run(init_op)

        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("checkpoint loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old checkpoint")
        if not output_testA:
            feature_output_list = []
        else:
            testa_out_list = []

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in tqdm(range(max_steps)):
                if coord.should_stop():
                    break
                out_reg, out_p5, out_fc1, out_fc2, out_fc3, label = sess.run([reg, r_p5, fc1, fc2, fc3, reg_labels],
                                                                             feed_dict={is_training: False,
                                                                                        keepprob: 1.0})
                out_p5 = np.reshape(out_p5, (batch_size * 8192))
                if not output_testA:
                    feature_output_list.append(
                        {"idx": step + 1, "pool5": out_p5, "fc1": out_fc1, "fc2": out_fc2, "fc3": out_fc3,
                         "label": label, "reg_out": out_reg})
                else:
                    testa_out_list.append(out_reg)

        except tf.errors.OutOfRangeError:
            print('Done training --epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)
        if not output_testA:
            joblib.dump(value=feature_output_list, filename=os.path.join(dirpath, "train_feature.pkl"), compress=3)
        else:
            joblib.dump(value=testa_out_list, filename=os.path.join(dirpath, "testA_out.pkl"), compress=3)


def test():
    train_batch_size = 1
    test_batch_size = 1
    max_epochs = 200
    max_steps = 1000000
    display_step = 1
    display_epoch = 1
    testa_out_list = []
    n_class = 47
    DTYPE = tf.float32
    DEBUG = False

    testfile = "../data/CIKM2017_testA/testA_Imp_3x3.pkl"
    testa_data, _ = read_data_from_pkl(datafile=testfile)
    dirpath = "../result/c3d_v1.0_batch_size:16_4_channel_fc_bn"
    CHECKPOINT_DIR = os.path.join(dirpath, "checkpoints")

    is_training = tf.placeholder_with_default(True, None)
    inputs = tf.placeholder(shape=(None, 15, 101, 101, 4), dtype=DTYPE)
    keepprob = tf.placeholder(dtype=DTYPE, shape=None)

    with tf.variable_scope("model") as scope:
        out_sm, out_reg, r_p5, fc1, fc2, fc3 = c3d_fc(inputs, keepprob, train_batch_size, n_class, DTYPE, is_training)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(tf.trainable_variables())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.device('/gpu:0'), tf.Session(config=config) as sess:

        sess.run(init_op)

        testa_out_list.clear()

        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("checkpoint loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old checkpoint")
        feature_output_list = []

        for i in tqdm(range(int(testa_data.shape[0] / test_batch_size))):
            batch_index_front = i * test_batch_size
            batch_index_end = (i + 1) * test_batch_size
            train_batch_x = testa_data[batch_index_front:batch_index_end]
            testa_out, out_p5, out_fc1, out_fc2, out_fc3 = sess.run([out_reg, r_p5, fc1, fc2, fc3],
                                                                    feed_dict={inputs: train_batch_x,
                                                                               is_training: False, keepprob: 1.0})
            # feature_output_list.append({"idx": i + 1, "pool5": out_p5, "fc1": out_fc1, "fc2":out_fc2, "fc3":out_fc3})
            testa_out_list.append(testa_out)

        # joblib.dump(value=feature_output_list, filename=os.path.join(dirpath,"testA_Feature.pkl"), compress=3)
        joblib.dump(value=testa_out_list, filename=os.path.join(dirpath, "testA_out.pkl"), compress=3)


if __name__ == "__main__":
    # train()
    fire.Fire()
    # output()
    # test()
