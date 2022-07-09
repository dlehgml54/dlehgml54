import tensorflow as tf

tf_v1 = tf.compat.v1
w_init = tf_v1.glorot_uniform_initializer()


def Conv(tensor, output_dim, k_size, rate=1, strides=1, padding='VALID', activation=None, bbias=True, name=None):
    return tf_v1.layers.conv2d(tensor, output_dim, [k_size, k_size], strides=[strides, strides], dilation_rate=rate,
                               padding=padding, activation=activation, kernel_initializer=w_init, use_bias=bbias,
                               name=name)


def Conv_T(tensor, output_dim, k_size, strides=1, activation=None, bbias=True, name=None, padding='VALID'):
    return tf_v1.layers.conv2d_transpose(tensor, output_dim, [k_size, k_size], strides=(strides, strides),
                                         padding=padding, activation=activation, use_bias=bbias, name=name, reuse=None)


def U_NET(x, num_class, name, reuse):
    with tf_v1.variable_scope(name) as GNet:
        if reuse:
            GNet.reuse_variables()

        # Contracting path
        x = tf_v1.nn.relu(Conv(x, output_dim=64, k_size=3, padding='SAME'))
        x = tf_v1.nn.relu(Conv(x, output_dim=64, k_size=3, padding='SAME'))
        f1 = x
        x = tf_v1.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')

        x = tf_v1.nn.relu(Conv(x, output_dim=128, k_size=3, padding='SAME'))
        x = tf_v1.nn.relu(Conv(x, output_dim=128, k_size=3, padding='SAME'))
        f2 = x
        x = tf_v1.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')

        x = tf_v1.nn.relu(Conv(x, output_dim=256, k_size=3, padding='SAME'))
        x = tf_v1.nn.relu(Conv(x, output_dim=256, k_size=3, padding='SAME'))
        f3 = x
        x = tf_v1.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')

        x = tf_v1.nn.relu(Conv(x, output_dim=512, k_size=3, padding='SAME'))
        x = tf_v1.nn.relu(Conv(x, output_dim=512, k_size=3, padding='SAME'))
        f4 = x
        x = tf_v1.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
        # ===== #

        x = tf_v1.nn.relu(Conv(x, output_dim=1024, k_size=3, padding='SAME'))
        x = tf_v1.nn.relu(Conv(x, output_dim=1024, k_size=3, padding='SAME'))

        # Expansive path
        x = Conv_T(x, output_dim=512, k_size=2, strides=2)
        x = tf.concat([f4, x], axis=3)
        x = tf_v1.nn.relu(Conv(x, output_dim=512, k_size=3, padding='SAME'))
        x = tf_v1.nn.relu(Conv(x, output_dim=512, k_size=3, padding='SAME'))

        x = Conv_T(x, output_dim=256, k_size=2, strides=2)
        x = tf.concat([f3, x], axis=3)
        x = tf_v1.nn.relu(Conv(x, output_dim=256, k_size=3, padding='SAME'))
        x = tf_v1.nn.relu(Conv(x, output_dim=256, k_size=3, padding='SAME'))

        x = Conv_T(x, output_dim=128, k_size=2, strides=2)
        x = tf.concat([f2, x], axis=3)
        x = tf_v1.nn.relu(Conv(x, output_dim=128, k_size=3, padding='SAME'))
        x = tf_v1.nn.relu(Conv(x, output_dim=128, k_size=3, padding='SAME'))

        x = Conv_T(x, output_dim=64, k_size=2, strides=2)
        x = tf.concat([f1, x], axis=3)
        x = tf_v1.nn.relu(Conv(x, output_dim=64, k_size=3, padding='SAME'))
        x = tf_v1.nn.relu(Conv(x, output_dim=64, k_size=3, padding='SAME'))

        x = Conv(x, output_dim=num_class, k_size=1, strides=1)

    return x
