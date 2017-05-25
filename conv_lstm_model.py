import tensorflow as tf
from conv_rnn_cell import ConvLSTMCell
from tensorflow.contrib.rnn import static_rnn, MultiRNNCell, stack_bidirectional_rnn


def _variable_with_weight_decay(name, shape, wd):
    # two gpu's train model
    # var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    var = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _conv2d(l_input, ksize, bsize, w_decay, padding='SAME'):
    w = _variable_with_weight_decay("weights", ksize, w_decay)
    b = _variable_with_weight_decay("biases", bsize, 0.0)
    return tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding=padding), b)


def _avg_pool(l_input, k):
    return tf.nn.avg_pool(l_input, ksize=[1, 2, 2, 1], strides=[1, k, k, 1], padding='SAME', name="pool")


def _batch_norm(l_input, is_training):
    return tf.contrib.layers.batch_norm(l_input, decay=0.9, epsilon=1e-5, center=True,
                                        scale=True, is_training=is_training, scope="bn")


def _fully_connected(l_input, wsize, bsize, w_decay):
    w = _variable_with_weight_decay("weights", wsize, wd=w_decay)
    # w = _variable_trun_normal_init("weights", weight_shape, 0.0, dtype=dtype)
    # b = _variable_constant_init("biases", bias_shape, 0.0, dtype=dtype)
    b = _variable_with_weight_decay("biases", bsize, wd=0.0)
    return tf.matmul(l_input, w) + b


def conv_lstm(input, time_step, dropout):
    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(input, time_step, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    # lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    # lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # conv_lstm cell
    shape = input.get_shape().as_list()[1:3]
    channel = input.get_shape().as_list()[-1]
    filter_n = 4 * channel
    kernel = [3, 3]
    layer_n = 2

    with tf.variable_scope("multi_layer_lstm"):
        cell = ConvLSTMCell(shape=shape, filters=filter_n, kernel=kernel)
        multi_cells = MultiRNNCell([cell] * layer_n)
    # Get lstm cell output

    outputs, final_state = static_rnn(multi_cells, x)

    '''====================5 layer conv network====================='''
    with tf.variable_scope("out_conv1"):
        conv_1 = _conv2d(outputs[-1], ksize=[3, 3, filter_n, 128], bsize=[128], w_decay=0.0005)
        relu_1 = tf.nn.relu(conv_1)
        avg_pool_1 = _avg_pool(relu_1, k=1)

    with tf.variable_scope("out_conv2"):
        conv_2 = _conv2d(avg_pool_1, ksize=[3, 3, 128, 256], bsize=[256], w_decay=0.0005)
        relu_2 = tf.nn.relu(conv_2)
        avg_pool_2 = _avg_pool(relu_2, k=2)

    with tf.variable_scope("out_conv3"):
        conv_3 = _conv2d(avg_pool_2, ksize=[3, 3, 256, 256], bsize=[256], w_decay=0.0005)
        relu_3 = tf.nn.relu(conv_3)
        avg_pool_3 = _avg_pool(relu_3, k=2)

    with tf.variable_scope("out_conv4"):
        conv_4 = _conv2d(avg_pool_3, ksize=[3, 3, 256, 256], bsize=[256], w_decay=0.0005)
        relu_4 = tf.nn.relu(conv_4)
        avg_pool_4 = _avg_pool(relu_4, k=2)

    with tf.variable_scope("out_conv5"):
        conv_5 = _conv2d(avg_pool_4, ksize=[3, 3, 256, 256], bsize=[256], w_decay=0.0005)
        relu_5 = tf.nn.relu(conv_5)
        avg_pool_5 = _avg_pool(relu_5, k=2)

    with tf.variable_scope("out_dense1"):
        # pool5 = tf.transpose(avg_pool_5, perm=[0, 1, 3, 2])
        pool5_re = tf.reshape(avg_pool_5, [-1, 6400])  # Reshape conv3 output to fit dense layer input
        fc1 = _fully_connected(pool5_re, wsize=[6400, 1024], bsize=[1024], w_decay=0.0005)
        relu_fc1 = tf.nn.relu(fc1)
        dropout_fc1 = tf.nn.dropout(relu_fc1, keep_prob=dropout)

    with tf.variable_scope("out_dense2"):
        fc2 = _fully_connected(dropout_fc1, wsize=[1024, 1024], bsize=[1024], w_decay=0.0005)
        relu_fc2 = tf.nn.relu(fc2)
        dropout_fc2 = tf.nn.dropout(relu_fc2, keep_prob=dropout)

    with tf.variable_scope("out_dense3"):
        fc3 = _fully_connected(dropout_fc2, wsize=[1024, 1], bsize=[1], w_decay=0.0005)

    return fc3, final_state
