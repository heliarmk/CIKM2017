import tensorflow as tf

'''
def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var
'''


def _variable_with_weight_decay(name, shape, wd, dtype):
    # two gpu's train model
    # var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())

    var = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=dtype)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_trun_normal_init(name, shape, l2_weights, dtype):
    return tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.1),
                           regularizer=tf.contrib.layers.l2_regularizer(l2_weights))


def _variable_constant_init(name, shape, l2_weights, dtype):
    return tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.1),
                           regularizer=tf.contrib.layers.l2_regularizer(l2_weights))


def conv3d(l_input, dtype, kernel_shape, bias_shape, w_decay, padding='SAME'):
    w = _variable_with_weight_decay("weights", kernel_shape, w_decay, dtype=dtype)
    b = _variable_with_weight_decay("biases", bias_shape, 0.0, dtype=dtype)
    return tf.nn.bias_add(tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding=padding), b)


def max_pool(name, l_input, k):
    return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)


def fully_connected(l_input, dtype, weight_shape, bias_shape, w_decay):
    w = _variable_with_weight_decay("weights", weight_shape, wd=w_decay, dtype=dtype)
    # w = _variable_trun_normal_init("weights", weight_shape, 0.0, dtype=dtype)
    # b = _variable_constant_init("biases", bias_shape, 0.0, dtype=dtype)
    b = _variable_with_weight_decay("biases", bias_shape, wd=0.0, dtype=dtype)
    return tf.matmul(l_input, w) + b


'''
def batch_norm(Ylogits, is_test, iteration, offset, convolutional=False):
    # adding the iteration prevents from averaging across non-existing iterations
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    bn_epsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2, 3])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bn_epsilon)
    return Ybn, update_moving_everages
'''

'''
def batch_norm(l_input, is_traininig, is_conv=True):
    bn_epsilon = 1e-3
    parm_shape = l_input.get_shape().as_list()[-1]
    with tf.variable_scope("batch_normalization"):
        beta = tf.get_variable(name="beta", shape=[parm_shape], initializer=tf.zeros_initializer(), trainable=True)
        gamma = tf.get_variable(name="gamma", shape=[parm_shape], initializer=tf.ones_initializer(), trainable=True)
        moving_mean = tf.get_variable(name="moving_mean", shape=[parm_shape], initializer=tf.zeros_initializer(),
                                      trainable=False)
        moving_var = tf.get_variable(name="moving_var", shape=[parm_shape], initializer=tf.ones_initializer(),
                                     trainable=False)

        if is_conv:
            moving_mean, moving_var = tf.nn.moments(l_input, [0, 1, 2, 3], name="moment")
        else:
            moving_mean, moving_var = tf.nn.moments(l_input, [0], name="moment")

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                ema_apply_op = ema.apply([moving_mean, moving_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(moving_mean), tf.identity(moving_var)

        mean, var = tf.cond(is_traininig, mean_var_with_update,
                            lambda: (ema.average(moving_mean), ema.average(moving_var)))
        normed = tf.nn.batch_normalization(l_input, mean, var, beta, gamma, bn_epsilon)
    return normed
'''

def batch_norm(l_input, is_training):
    return tf.contrib.layers.batch_norm(l_input, decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,is_training=is_training,scope = "bn")

def c3d_fc(_input, _dropout, batch_size, n_classs, dtype, is_training):
    # Convolution Layer 1
    with tf.variable_scope("conv1"):
        conv1 = conv3d(_input, dtype, [3, 3, 3, 4, 64], [64], 0.0005)
        bn1 = batch_norm(conv1, is_training)
        relu1 = tf.nn.relu(bn1, 'relu')
        pool1 = max_pool('pooling', relu1, k=1)

    # Convolution Layer 2
    with tf.variable_scope("conv2"):
        conv2 = conv3d(pool1, dtype, [3, 3, 3, 64, 128], [128], 0.0005)
        bn2 = batch_norm(conv2, is_training)
        relu2 = tf.nn.relu(bn2, 'relu')
        pool2 = max_pool('pooling', relu2, k=2)

    # Convolution Layer 3
    with tf.variable_scope("conv3a"):
        conv3a = conv3d(pool2, dtype, [3, 3, 3, 128, 256], [256], 0.0005)
        bn3a = batch_norm(conv3a, is_training)
        relu3a = tf.nn.relu(bn3a, "relu")
    with tf.variable_scope("conv3b"):
        conv3b = conv3d(relu3a, dtype, [3, 3, 3, 256, 256], [256], 0.0005)
        bn3b = batch_norm(conv3b,is_training)
        relu3b = tf.nn.relu(bn3b, "relu")
        pool3 = max_pool("pooling", relu3b, k=2)

    # Convolution Layer 4
    with tf.variable_scope("conv4a"):
        conv4a = conv3d(pool3, dtype, [3, 3, 3, 256, 512], [512], 0.0005)
        bn4a = batch_norm(conv4a,is_training)
        relu4a = tf.nn.relu(bn4a, "relu")
    with tf.variable_scope("conv4b"):
        conv4b = conv3d(relu4a, dtype, [3, 3, 3, 512, 512], [512], 0.0005)
        bn4b = batch_norm(conv4b, is_training)
        relu4b = tf.nn.relu(bn4b, 'relu')
        pool4 = max_pool('pooling', relu4b, k=2)

    # Convolution Layer 5
    with tf.variable_scope("conv5a"):
        conv5a = conv3d(pool4, dtype, [3, 3, 3, 512, 512], [512], 0.0005)
        bn5a = batch_norm(conv5a,is_training)
        relu5a = tf.nn.relu(bn5a, "relu")
    with tf.variable_scope("conv5b"):
        conv5b = conv3d(relu5a, dtype, [3, 3, 3, 512, 512], [512], 0.0005)
        bn5b = batch_norm(conv5b,is_training)
        relu5b = tf.nn.relu(bn5b, 'relu')
        pool5 = max_pool('pooling', relu5b, k=2)

    # Fully connected layer
    with tf.variable_scope("dense1"):
        pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
        pool5_re = tf.reshape(pool5, [batch_size, 8192])  # Reshape conv3 output to fit dense layer input
        fc1 = fully_connected(pool5_re, dtype, [8192, 4096], [4096], 0.0005)
        bnfc1 = batch_norm(fc1, is_training)
        # dense1 = tf.reshape(pool5, [batch_size, 8192])  # Reshape conv3 output to fit dense layer input
        # dense1 = fully_connected(dense1, dtype, [8192, 2048], [2048], 0.0005)

        relufc1 = tf.nn.relu(bnfc1, name='relu')  # Relu activation
        dropoutfc1 = tf.nn.dropout(relufc1, _dropout, name="dropout")

    with tf.variable_scope("dense2"):
        fc2 = fully_connected(dropoutfc1, dtype, [4096, 4096], [4096], 0.0005)
        bnfc2 = batch_norm(fc2, is_training)
        # dense2 = fully_connected(dense1, dtype, [2048, 1024], [1024], 0.0005)
        relufc2 = tf.nn.relu(bnfc2, name='relu')  # Relu activation
        dropoutfc2 = tf.nn.dropout(relufc2, _dropout, name="dropout")

    with tf.variable_scope("dense3"):
        fc3 = fully_connected(dropoutfc2, dtype, [4096, 4096], [4096], 0.0005)
        bnfc3 = batch_norm(fc3, is_training)
        # dense2 = fully_connected(dense1, dtype, [2048, 1024], [1024], 0.0005)
        relufc3 = tf.nn.relu(bnfc3, name='relu')  # Relu activation
        dropoutfc3 = tf.nn.dropout(relufc3, _dropout, name="dropout")

    # Output: class prediction
    with tf.variable_scope("output_class"):
        out_softmax = fully_connected(dropoutfc3, dtype, [4096, n_classs], [n_classs], 0.0005)
        # out = fully_connected(dense2, dtype, [1024, 1], [1], 0.0005)
    with tf.variable_scope("output_num"):
        out_regression = fully_connected(dropoutfc3, dtype, [4096, 1], [1], 0.0005)

    return out_softmax, out_regression, pool5_re, fc1, fc2, fc3



def c3d_fcn(_input, _dropout, batch_size, n_classs, dtype, is_training):
    # Convolution Layer 1
    with tf.variable_scope("conv1"):
        conv1 = conv3d(_input, dtype, [3, 3, 3, 4, 64], [64], 0.0005)
        bn1 = batch_norm(conv1, is_training)
        relu1 = tf.nn.relu(bn1, 'relu')
        pool1 = max_pool('pooling', relu1, k=1)

    # Convolution Layer 2
    with tf.variable_scope("conv2"):
        conv2 = conv3d(pool1, dtype, [3, 3, 3, 64, 128], [128], 0.0005)
        bn2 = batch_norm(conv2, is_training)
        relu2 = tf.nn.relu(bn2, 'relu')
        pool2 = max_pool('pooling', relu2, k=2)

    # Convolution Layer 3
    with tf.variable_scope("conv3a"):
        conv3a = conv3d(pool2, dtype, [3, 3, 3, 128, 256], [256], 0.0005)
        bn3a = batch_norm(conv3a, is_training)
        relu3a = tf.nn.relu(bn3a, "relu")
    with tf.variable_scope("conv3b"):
        conv3b = conv3d(relu3a, dtype, [3, 3, 3, 256, 256], [256], 0.0005)
        bn3b = batch_norm(conv3b, is_training)
        relu3b = tf.nn.relu(bn3b, "relu")
        pool3 = max_pool("pooling", relu3b, k=2)

    # Convolution Layer 4
    with tf.variable_scope("conv4a"):
        conv4a = conv3d(pool3, dtype, [3, 3, 3, 256, 512], [512], 0.0005)
        bn4a = batch_norm(conv4a, is_training)
        relu4a = tf.nn.relu(bn4a, "relu")
    with tf.variable_scope("conv4b"):
        conv4b = conv3d(relu4a, dtype, [3, 3, 3, 512, 512], [512], 0.0005)
        bn4b = batch_norm(conv4b, is_training)
        relu4b = tf.nn.relu(bn4b, 'relu')
        pool4 = max_pool('pooling', relu4b, k=2)

    # Convolution Layer 5
    with tf.variable_scope("conv5a"):
        conv5a = conv3d(pool4, dtype, [3, 3, 3, 512, 512], [512], 0.0005)
        bn5a = batch_norm(conv5a, is_training)
        relu5a = tf.nn.relu(bn5a, "relu")
    with tf.variable_scope("conv5b"):
        conv5b = conv3d(relu5a, dtype, [3, 3, 3, 512, 512], [512], 0.0005)
        bn5b = batch_norm(conv5b, is_training)
        relu5b = tf.nn.relu(bn5b, 'relu')
        pool5 = max_pool('pooling', relu5b, k=2)

    # replace the fully connected dense1 and dense2
    with tf.variable_scope("fc6-conv"):
        conv6 = conv3d(pool5, dtype, [1, 4, 4, 512, 4096], [4096], 0.0005, "VALID")
        bn6 = batch_norm(conv6, is_training)
        relu6 = tf.nn.relu(bn6, "relu")
        dropout6 = tf.nn.dropout(relu6, _dropout, name="dropout")

    with tf.variable_scope("fc7-conv"):
        conv7 = conv3d(dropout6, dtype, [1, 1, 1, 4096, 4096], [4096], 0.0005, "VALID")
        bn7 = batch_norm(conv7, is_training)
        relu7 = tf.nn.relu(bn7, "relu")
        dropout7 = tf.nn.dropout(relu7, _dropout, name="dropout")

    with tf.variable_scope("output"):
        out = conv3d(dropout7, dtype, [1, 1, 1, 4096, n_classs], [n_classs], 0.0000, "VALID")
        out = tf.transpose(out, perm=[0, 1, 4, 2, 3])
        out = tf.reshape(out, [batch_size, n_classs])

    with tf.variable_scope("output_num"):
        out_num = conv3d(dropout7, dtype, [1,1,1,4096, 1], [1], 0.0000, "VALID")
        out_num = tf.transpose(out_num, perm=[0, 1, 4, 2, 3])
        out_num = tf.reshape(out_num, [batch_size, 1])

    return out, pool5, out_num
    #return out, pool5