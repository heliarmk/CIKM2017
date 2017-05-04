import tensorflow as tf

'''
def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var
'''


def _variable_with_weight_decay(name, shape, wd):
    # two gpu's train model
    # var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())

    var = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(1.0))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def conv3d(l_input, kernel_shape, bias_shape, w_decay):
    w = _variable_with_weight_decay("weights", kernel_shape, w_decay)
    b = _variable_with_weight_decay("biases", bias_shape, 0.0)
    return tf.nn.bias_add(tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'), b)


def max_pool(name, l_input, k):
    return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)


def fully_connected(l_input, weight_shape, bias_shape, w_decay):
    w = _variable_with_weight_decay("weights", weight_shape, w_decay)
    b = _variable_with_weight_decay("biases", bias_shape, 0.0)
    return tf.matmul(l_input, w) + b


def inference_c3d(_input, _dropout, batch_size):
    # Convolution Layer 1
    with tf.variable_scope("conv1"):
        conv1 = conv3d(_input, [3, 3, 3, 4, 64], [64], 0.0005)
        relu1 = tf.nn.relu(conv1, 'relu')
        pool1 = max_pool('pooling', relu1, k=1)

    # Convolution Layer 2
    with tf.variable_scope("conv2"):
        conv2 = conv3d(pool1, [3, 3, 3, 64, 128], [128], 0.0005)
        relu2 = tf.nn.relu(conv2, 'relu')
        pool2 = max_pool('pooling', relu2, k=2)

    # Convolution Layer 3
    with tf.variable_scope("conv3a"):
        conv3a = conv3d(pool2, [3, 3, 3, 128, 256], [256], 0.0005)
        relu3a = tf.nn.relu(conv3a, "relu")
    with tf.variable_scope("conv3b"):
        conv3b = conv3d(relu3a, [3, 3, 3, 256, 256], [256], 0.0005)
        relu3b = tf.nn.relu(conv3b, "relu")
        pool3 = max_pool("pooling", relu3b, k=2)

    # Convolution Layer 4
    with tf.variable_scope("conv4a"):
        conv4a = conv3d(pool3, [3, 3, 3, 256, 512], [512], 0.0005)
        relu4a = tf.nn.relu(conv4a, "relu")
    with tf.variable_scope("conv4b"):
        conv4b = conv3d(relu4a, [3, 3, 3, 512, 512], [512], 0.0005)
        relu4b = tf.nn.relu(conv4b, 'relu')
        pool4 = max_pool('pooling', relu4b, k=2)

    # Convolution Layer 5
    with tf.variable_scope("conv5a"):
        conv5a = conv3d(pool4, [3, 3, 3, 512, 512], [512], 0.0005)
        relu5a = tf.nn.relu(conv5a, "relu")
    with tf.variable_scope("conv5b"):
        conv5b = conv3d(relu5a, [3, 3, 3, 512, 512], [512], 0.0005)
        relu5b = tf.nn.relu(conv5b, 'relu')
        pool5 = max_pool('pooling', relu5b, k=2)

    # Fully connected layer
    with tf.variable_scope("dense1"):
        pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
        #dense1 = tf.reshape(pool5, [batch_size, 8192])  # Reshape conv3 output to fit dense layer input
        #dense1 = fully_connected(dense1, [8192, 4096], [4096], 0.0005)
        dense1 = tf.reshape(pool5, [batch_size, 8192])  # Reshape conv3 output to fit dense layer input
        dense1 = fully_connected(dense1, [8192, 2048], [2048], 0.0005)

        dense1 = tf.nn.relu(dense1, name='relu')  # Relu activation
        dense1 = tf.nn.dropout(dense1, _dropout, name="dropout")

    with tf.variable_scope("dense2"):
        #dense2 = fully_connected(dense1, [4096, 4096], [4096], 0.0005)
        dense2 = fully_connected(dense1, [2048, 1024], [1024], 0.0005)
        dense2 = tf.nn.relu(dense2, name='relu')  # Relu activation
        dense2 = tf.nn.dropout(dense2, _dropout, name="dropout")

    # Output: class prediction
    with tf.variable_scope("output"):
        #out = fully_connected(dense2, [4096, 1], [1], 0.0005)
        out = fully_connected(dense2, [1024, 1], [1], 0.0)

    return out
