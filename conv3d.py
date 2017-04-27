import tensorflow as tf
import numpy as np
import joblib
import time
import os

FC_SIZE = 1024
DTYPE = tf.float32


def _weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, dtype=DTYPE, initializer=tf.truncated_normal_initializer(stddev=0.1),
                           regularizer=tf.contrib.layers.l2_regularizer(1.0))

def _bias_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, dtype=DTYPE, initializer=tf.constant_initializer(0.1, dtype=DTYPE),
                           regularizer=tf.contrib.layers.l2_regularizer(1.0))


def conv3dnet(input, keepprob):
    prev_layer = input

    in_filters = 4
    out_class = 1
    with tf.variable_scope('conv1') as scope:
        out_filters = 16
        kernel = _weight_variable('weights', [3, 3, 3, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

        prev_layer = conv1
        in_filters = out_filters

    pool1 = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    norm1 = pool1  # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')

    prev_layer = norm1

    with tf.variable_scope('conv2') as scope:
        out_filters = 32
        kernel = _weight_variable('weights', [3, 3, 3, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

        prev_layer = conv2
        in_filters = out_filters

    # normalize prev_layer here
    prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    with tf.variable_scope('conv3_1') as scope:
        out_filters = 64
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias, name=scope.name)
        in_filters = out_filters
    '''
    with tf.variable_scope('conv3_2') as scope:
        out_filters = 64
        kernel = _weight_variable('weights', [3, 3, 3, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias, name=scope.name)
        in_filters = out_filters
    with tf.variable_scope('conv3_3') as scope:
        out_filters = 64
        kernel = _weight_variable('weights', [3, 3, 3, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias, name=scope.name)
        in_filters = out_filters
    '''
    # normalize prev_layer here
    prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    with tf.variable_scope('local3') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])
        # local3 = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)
        local3 = tf.nn.relu(tf.nn.dropout(tf.matmul(prev_layer_flat, weights) + biases, keepprob), name=scope.name)

    prev_layer = local3
    with tf.variable_scope('local4') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])
        local4 = tf.nn.relu(tf.nn.dropout(tf.matmul(prev_layer_flat, weights) + biases, keepprob), name=scope.name)

    prev_layer = local4

    with tf.variable_scope('softmax_linear') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        weights = _weight_variable('weights', [dim, out_class])
        biases = _bias_variable('biases', [1])
        softmax_linear = tf.add(tf.matmul(prev_layer, weights), biases, name=scope.name)

    return softmax_linear


# def loss(logits, labels):
#    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
#        logits, labels, name='cross_entropy_per_example')

#    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def calc_loss(preds, labels):
    return tf.reduce_mean(tf.square(preds - labels))


def read_data(datafile):
    datas = joblib.load(datafile)
    inputs, labels = [], []
    for data in datas:
        inputs.append(data["input"])
        labels.append(data["label"])

    inputs = np.array(inputs).reshape(-1, 15, 101, 101, 4)
    labels = np.array(labels).reshape(-1, 1)

    return inputs, labels


def train():
    train_batch_size = 10
    test_batch_size = 1
    epochs = 100
    display_step = 1
    fold = 10
    testa_out_list = []
    store_result = True
    save_model = True

    #dirpath = "./" + "FC_SIZE:" + str(FC_SIZE) + "_batch_size" + str(train_batch_size) + "_data_agg" + "_10_fold_cv"
    dirpath = "../result/" + "FC_SIZE:" + str(FC_SIZE) + "_batch_size" + str(train_batch_size) + "_10_fold_cv" + "_4_channel"
    #read data
    #trainfile = "../data/CIKM2017_train/train_Imp_3x3_mean_axis1_ori_and_flip_ax1&2_shuffle.pkl"
    trainfile = "../data/CIKM2017_train/train_Imp_3x3.pkl"
    testfile = "../data/CIKM2017_testA/testA_Imp_3x3.pkl"

    train_data, train_label = read_data(datafile=trainfile)
    testa_data, _ = read_data(datafile=testfile)
    
    fold_step = int(train_data.shape[0] / fold)
    
    inputs = tf.placeholder(tf.int16, shape=(None, 15, 101, 101, 4))
    labels = tf.placeholder(tf.int16, shape=(None, 1))
    keepprob = tf.placeholder(DTYPE)

    preds = conv3dnet(inputs, keepprob)
    loss = calc_loss(preds, labels)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    # rmse_test = tf.square(preds - labels)

    init = tf.initialize_all_variables()
    #param saver
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.device('/gpu:0'), tf.Session(config=config) as sess:
        for n in range(fold):
            sess.run(init)
            # split dataset
            test_index_begin = n * fold_step
            test_index_end = (n + 1) * fold_step
            x_train = np.delete(train_data, np.arange(test_index_begin, test_index_end), 0)
            y_train = np.delete(train_label, np.arange(test_index_begin, test_index_end), 0)
            x_test = train_data[test_index_begin:test_index_end]
            y_test = train_label[test_index_begin:test_index_end]
            for epoch in range(epochs):
                avg_loss = 0
                avg_rmse = 0
                testa_out_list.clear()
                total_train_batch = int(x_train.shape[0] / train_batch_size)
                for i in range(total_train_batch):
                    batch_index_front = i * train_batch_size
                    batch_index_end = (i + 1) * train_batch_size
                    train_batch_x = x_train[batch_index_front:batch_index_end]
                    train_batch_y = y_train[batch_index_front:batch_index_end]
                    _, l = sess.run([optimizer, loss], feed_dict={inputs: train_batch_x, labels: train_batch_y, keepprob:0.5})
                    avg_loss += train_batch_size * l / x_train.shape[0]

                for i in range(int(fold_step / test_batch_size)):
                    batch_index_front = i * test_batch_size
                    batch_index_end = (i + 1) * test_batch_size
                    test_batch_x = x_test[batch_index_front:batch_index_end]
                    test_batch_y = y_test[batch_index_front:batch_index_end]
                    rmse = sess.run(loss, feed_dict={inputs: test_batch_x, labels: test_batch_y,keepprob:1.0})
                    avg_rmse += test_batch_size * rmse / x_test.shape[0]

                # output testA pred
                for i in range(int(testa_data.shape[0] / test_batch_size)):
                    batch_index_front = i * test_batch_size
                    batch_index_end = (i + 1) * test_batch_size
                    train_batch_x = testa_data[batch_index_front:batch_index_end]
                    testa_out = sess.run(preds, feed_dict={inputs: train_batch_x,keepprob:1.0})
                    testa_out_list.append(testa_out)
                    # avg_rmse += rmse / x_test.shape[0]

                rmse_train = np.sqrt(avg_loss)
                rmse_test = np.sqrt(avg_rmse)
                result = "Epoch:%03d,Fold:%02d,Avg_loss=%9f,avg_rmse=%9f" % (
                epoch, n, rmse_train, rmse_test)

                if epoch % display_step == 0:
                    print(result)
                # if save_model == True:

                if store_result == True:
                    if not os.path.isdir(dirpath):
                        os.mkdir(dirpath)
                    filename = dirpath + "/time:" + time.ctime() +"_fold:" + str(n) +"_epoch:" + str(epoch)\
                               + "_loss:" + str(rmse_train) +"_rmse" + str(rmse_test) + ".pkl"
                    output_dict = {"restlt": result, "output": testa_out_list}
                    joblib.dump(value=output_dict, filename=filename,compress=3)
                    #print("Restlt save in file %s" % filename)
                if save_model:
                    model_path = dirpath + "/model"
                    if not os.path.isdir(model_path):
                        os.mkdir(model_path)
                    save_path = model_path + "/time:" + time.ctime() +"_fold:" + str(n) +"_epoch:" + str(epoch)\
                               + "_loss:" + str(rmse_train) +"_rmse" + str(rmse_test) +".ckpt"
                    saver.save(sess=sess,save_path=save_path)
                    #print("Model save in file %s" % save_path)

if __name__ == "__main__":
    train()
