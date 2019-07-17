import tensorflow as tf
import numpy as np
import config as Con

def conv2d(name, input, dim, bn=True, is_pooling=False):
    filter_w = tf.Variable(tf.random_normal([Con.FILTER_SIZE, Con.FILTER_SIZE, input.get_shape().as_list()[3], dim],
                                            stddev=Con.CONV2D_STDDEV), name=name)
    conv = tf.nn.conv2d(input, filter_w, strides=[1, 1, 1, 1], padding="SAME")
    conv = tf.layers.batch_normalization(conv, center=True, scale=True, training=bn)
    conv = tf.nn.relu(conv)

    before_pooling = conv
    if is_pooling:
        conv = max_pool_2by2(conv)

    return conv, before_pooling


def de_conv2d(name, input, add_chanels, bn=True):
    filter_de_w = tf.Variable(
        tf.random_normal([Con.DE_FILTER_SIZE, Con.DE_FILTER_SIZE, input.get_shape().as_list()[-1] // 2,
                          input.get_shape().as_list()[-1]], stddev=Con.DE_CONV2D_STDDEV, name=name))
    de_conv = tf.nn.conv2d_transpose(input, filter_de_w,
                                     output_shape=[input.get_shape().as_list()[0], input.get_shape().as_list()[1] * 2,
                                                   input.get_shape().as_list()[2] * 2,
                                                   input.get_shape().as_list()[3] // 2],
                                     strides=[1, 2, 2, 1], padding="SAME")
    de_conv = tf.layers.batch_normalization(de_conv, center=True, scale=True, training=bn)
    de_conv = tf.nn.relu(de_conv)

    if add_chanels != None:
        de_conv = tf.concat([de_conv, add_chanels], axis=3)

    return de_conv


def final_lay(name, input):
    filter_w = tf.Variable(tf.random_normal([Con.FILTER_SIZE, Con.FILTER_SIZE, input.get_shape().as_list()[3], 1],
                                            stddev=Con.CONV2D_STDDEV), name=name)
    conv = tf.nn.conv2d(input, filter_w, strides=[1, 1, 1, 1], padding="SAME")

    conv = tf.nn.sigmoid(conv)
    return conv


def max_pool_2by2(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
