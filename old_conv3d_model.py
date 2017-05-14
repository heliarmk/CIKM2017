import tensorflow as tf
import numpy as np

FC_SIZE = 1024
DTYPE = tf.float32

def _weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, dtype=DTYPE, initializer=tf.truncated_normal_initializer(stddev=0.1),
                           regularizer=tf.contrib.layers.l2_regularizer(1.0))

def _bias_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, dtype=DTYPE, initializer=tf.constant_initializer(0.1, dtype=DTYPE),
                           regularizer=tf.contrib.layers.l2_regularizer(1.0))

def conv3dnet(input, keepprob, out_class):
    prev_layer = input

    in_filters = 4
    out_class = out_class
    with tf.variable_scope('conv1') as scope:
        out_filters = 128
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
        out_filters = 64
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
        kernel = _weight_variable('weights', [3, 3, 3, in_filters, out_filters])
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
    '''
    with tf.variable_scope('local4') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])
        local4 = tf.nn.relu(tf.nn.dropout(tf.matmul(prev_layer_flat, weights) + biases, keepprob), name=scope.name)

    prev_layer = local4
    '''

    with tf.variable_scope('softmax_linear') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        weights = _weight_variable('weights', [dim, out_class])
        biases = _bias_variable('biases', [1])
        softmax_linear = tf.add(tf.matmul(prev_layer, weights), biases, name=scope.name)

    return softmax_linear