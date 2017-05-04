import tensorflow as tf
import numpy as np
import joblib
import time
import os
from old_conv3d_model import conv3dnet
from c3d_model import inference_c3d


# def loss(logits, labels):
#    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
#        logits, labels, name='cross_entropy_per_example')

#    return tf.reduce_mean(cross_entropy, name='xentropy_mean')
def get_num_records(tf_record_file):
  return len([x for x in tf.python_io.tf_record_iterator(tf_record_file)])

def calc_loss(preds, labels):
    return tf.reduce_mean(tf.square(preds - labels))

def read_data_from_pkl(datafile):
    datas = joblib.load(datafile)
    datas = np.random.permutation(datas)
    inputs, labels = [], []
    for data in datas:
        inputs.append(data["input"])
        labels.append(data["label"])

    inputs = np.array(inputs).astype(np.float32).reshape(-1, 15, 101, 101, 4)
    labels = np.array(labels).astype(np.float32).reshape(-1, 1)

    return inputs, labels


def read_and_decode_from_tfrecord(trainfilename, validfilename, is_training, batch_size, shuffle=False):

    # 根据文件名生成一个队列
    train_queue = tf.train.string_input_producer([trainfilename])
    valid_queue = tf.train.string_input_producer([validfilename])

    # 挑选文件队列，实现training的过程中测试
    queue_select = tf.cond(is_training,
                           lambda: tf.constant(0),
                           lambda: tf.constant(1))
    queue = tf.QueueBase.from_list(queue_select, [train_queue, valid_queue])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.float32),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'depth': tf.FixedLenFeature([], tf.int64),
                                           'channel': tf.FixedLenFeature([], tf.int64)
                                       })
    img = tf.decode_raw(features['image_raw'], tf.int16)
    depth, height, width, channel = 15, 101, 101, 4
    img = tf.reshape(img, [depth, height, width, channel])
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = features['label']
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                      batch_size=batch_size, capacity=2000, min_after_dequeue=1000,
                                                      num_threads=2)
    else:
        image_batch, label_batch = tf.train.batch([img, label],
                                                  batch_size=batch_size, capacity=2000, num_threads=2)

    return tf.cast(image_batch,tf.float32), tf.cast(label_batch, tf.float32)


def train():
    train_batch_size = 20
    test_batch_size = 1
    #max_epochs = 200
    max_steps = 1000000
    train_set_num = 54000
    valid_set_num = 6000
    display_step = 100
    testa_out_list = []
    store_result = True
    SNAPSTEP = 1000

    #dirpath = "./" + "FC_SIZE:" + str(FC_SIZE) + "_batch_size" + str(train_batch_size) + "_data_agg" + "_10_fold_cv"
    dirpath = "../result/" + "c3d_v1.0_batch_size:" + str(train_batch_size) +  "_4_channel" +"_agg"
    #dirpath = "../result/" + "old_conv3d_batch_size:" + str(train_batch_size) +  "_4_channel" +"_agg"

    log_dir = os.path.join(dirpath,"log")
    # read data
    # trainfile = "../data/CIKM2017_train/train_Imp_3x3_mean_axis1_ori_and_flip_ax1&2_shuffle.pkl"
    # trainfile = "../data/CIKM2017_train/train_Imp_3x3.pkl"
    trainfile = "../data/CIKM2017_train/train_Imp_3x3_fliped&rotated.tfrecords"
    #trainfile = "../data/CIKM2017_train/train_Imp_3x3_fliped.pkl"
    #trainfile = "../data/CIKM2017_train/train_Imp_3x3.tfrecords"
    validfile = "../data/CIKM2017_train/valid_Imp_3x3_fliped&rotated.tfrecords"
    #validfile = "../data/CIKM2017_train/valid_Imp_3x3.tfrecords"
    testfile = "../data/CIKM2017_testA/testA_Imp_3x3.pkl"

    #train_set_num = get_num_records(trainfile)
    #valid_set_num = get_num_records(validfile)

    print(train_set_num)
    print(valid_set_num)

    #base_data, base_label = read_data_from_pkl(datafile=trainfile)

    is_training = tf.placeholder_with_default(True, None)
    data, label = read_and_decode_from_tfrecord(trainfile, validfile, is_training, train_batch_size, shuffle=True)
    testa_data, _ = read_data_from_pkl(datafile=testfile)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # fold_step = int(train_data.shape[0] / fold)

    inputs = tf.placeholder(tf.float32, shape=(None, 15, 101, 101, 4))
    # labels = tf.placeholder(tf.float32, shape=(None, 1))
    keepprob = tf.placeholder_with_default(0.5,None)

    # preds = conv3dnet(inputs, keepprob)
    with tf.variable_scope("model") as scope:
        preds = inference_c3d(data, keepprob, train_batch_size)
        #preds = conv3dnet(data, keepprob)
        scope.reuse_variables()
        #testa_output = conv3dnet(inputs, keepprob)
        testa_output = inference_c3d(inputs, keepprob, test_batch_size)

    loss = calc_loss(preds, label)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       100000, 0.96, staircase=True)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    summary_op = tf.summary.merge_all()
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    # param saver
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.device('/gpu:0'), tf.Session(config=config) as sess:

        train_writer = tf.summary.FileWriter(logdir=os.path.join(log_dir, "train"), graph=sess.graph)
        val_writer = tf.summary.FileWriter(logdir=os.path.join(log_dir, "test"))

        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_loss = 0
        epoch = 0
        try:
            for step in range(max_steps):
                if coord.should_stop():
                    break

                _, l = sess.run([train_op, loss], feed_dict={is_training: True, keepprob: 0.5})
                #avg_loss += l / train_num
                train_loss += l * train_batch_size
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, global_step=step)

                if step % display_step == 0 :
                    print("Step:%05d,loss:%9f" %(step, np.sqrt(l)))

                if (step * train_batch_size) % train_set_num == 0 and not step == 0:
                    testa_out_list.clear()
                    epoch += 1
                    valid_rmse = 0
                    for i in range(int(valid_set_num / train_batch_size)):
                        rmse = sess.run(loss, feed_dict={is_training: False, keepprob: 1.0})
                        valid_rmse += rmse * train_batch_size

                    train_loss /= train_set_num
                    valid_rmse /= valid_set_num

                    train_loss = np.sqrt(train_loss)
                    valid_rmse = np.sqrt(valid_rmse)

                    result = "Epoch:%03d,Avg_loss=%9f,avg_rmse=%9f" % (epoch, train_loss, valid_rmse)
                    print(result)

                    # output testA predicts
                    for i in range(int(testa_data.shape[0] / test_batch_size)):
                        batch_index_front = i * test_batch_size
                        batch_index_end = (i + 1) * test_batch_size
                        train_batch_x = testa_data[batch_index_front:batch_index_end]
                        out = sess.run(testa_output, feed_dict={inputs: train_batch_x, keepprob: 1.0})
                        testa_out_list.append(out)
                    # avg_rmse += rmse / x_test.shape[0]

                    if store_result == True:
                        if not os.path.isdir(dirpath):
                            os.mkdir(dirpath)
                        filename = dirpath + "/time:" + time.ctime() + "epoch:" + str(epoch) \
                           + "_loss:" + str(train_loss) + "_rmse:" + str(valid_rmse) + ".pkl"
                        output_dict = {"rmse": valid_rmse, "loss": train_loss, "output": testa_out_list}
                        joblib.dump(value=output_dict, filename=filename, compress=3)

                    train_loss = 0
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str, global_step=step)

                # print("Restlt save in file %s" % filename)
                if step % SNAPSTEP == 0:
                    checkpoint_path = os.path.join(log_dir, "train/model.ckpt")
                    saver.save(sess, checkpoint_path, global_step=step)
                # print("Model save in file %s" % save_path)
        except tf.errors.OutOfRangeError:
            print('Done training --epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)
        '''
        train_data = base_data[3000:]
        train_label = base_label[3000:]
        valid_data = base_data[0:3000]
        valid_label = base_label[0:3000]

        for epoch in range(epochs):
            
            loss = 0
            rmse = 0
            testa_out_list.clear()
            train_batch_num = int(train_data.shape[0]/ train_batch_size)
            for i in range(train_batch_num):
                batch_index_front = i * train_batch_size
                batch_index_end = (i+1) * train_batch_size
                train_data_batch = train_data[batch_index_front:batch_index_end]
                train_label_batch = train_label[batch_index_front:batch_index_end]
                _ , l = sess.run([train_op, loss], feed_dict = {inputs: train_data_batch, labels: train_label_batch, keepprob: 0.5})
                loss += l * train_batch_size 
            
            valid_batch_num = int(valid_data.shape[0] / train_batch_size)
            for i in range(valid_batch_num):
                batch_index_front = i * train_set_size
                batch_index_end = (i+1) * train_batch_size
                valid_data_batch = valid_data[batch_index_front:batch_index_end]
                valid_label_batch = valid_label[batch_index_front:batch_index_end]
                r = sess.run(loss, feed_dict = {inputs: valid_data_batch, labels: valid_label_batch, keepprob :1.0})
                rmse += r * train_batch_size

            #output testa
            for i in range(testa_data.shape[0]):
                output_batch = testa_data[i]
                testa_out = sess.run(preds, feed_dict = {inputs:output_batch, keepprob:1.0})
                testa_out_list.append(testa_out)

            loss = np.sqrt(loss)
            rmse = np.sqrt(rmse)

            result = "Epoch:%03d, loss=%9f, rmse=%9f" % (epoch, loss, rmse)

            if epoch % display_epoch == 0:
                print(result)
            
            if store_result:
                if not os.path.isdir(dirpath):
                    os.mkdir(dirpath)
                filename = dirpath + "/Epoch:" + str(epoch) + "_loss:" + str(loss) +"_rmse" + str(rmse) + "_time:" + time.ctime() + ".pkl"
                output_dict = {"restlt": result, "output": testa_out_list}
                joblib.dump(value=output_dict, filename=filename,compress=3)
            '''
if __name__ == "__main__":
    train()
