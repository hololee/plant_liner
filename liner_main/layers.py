import numpy as np
import config as Con


class layers():
    CONV_NORMAL = 0
    CONV_DILATED = 1
    CONV_ASYMMETRIC = 2

    def __init__(self, tf):
        self._tf = tf

    def initial(self, name, input, dim=13, bn=True):
        print("[INPUT] : {}".format(input.get_shape().as_list()))

        pooling = self._tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        filter = self._tf.Variable(
            self._tf.random_normal([3, 3, input.get_shape().as_list()[3], dim], stddev=Con.INITIAL_STDDEV, name=name))
        conv = self._tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding="SAME")
        conv = self._tf.layers.batch_normalization(conv, center=True, scale=True, training=bn)
        conv = self._tf.concat([pooling, conv], axis=3)

        print("[{}] : {}".format(name, conv.get_shape().as_list()))
        return conv

    def conv2d(self, name, input, dim, name_1by1_reduce, name_1by1_expend, dim_1by1, bn=True, is_pooling=False,
               conv_style={"conv": CONV_NORMAL, "param": 0}):
        ## pooling part.
        pooling = self._tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        ##reduce dim
        filter_1by1_reduce = self._tf.Variable(
            self._tf.random_normal(shape=[1, 1, input.get_shape().as_list()[3], dim_1by1], stddev=Con.CONV2D_STDDEV,
                                   name=name_1by1_reduce))
        mid_conv = self._tf.nn.conv2d(input, filter_1by1_reduce, strides=[1, 1, 1, 1], padding="SAME")
        mid_conv = self._tf.layers.batch_normalization(mid_conv, center=True, scale=True, training=bn)
        mid_conv = self._tf.nn.relu(mid_conv)

        ##convolution
        if conv_style["conv"] == 0:
            filter_w = self._tf.Variable(
                self._tf.random_normal([Con.FILTER_SIZE, Con.FILTER_SIZE, mid_conv.get_shape().as_list()[3], dim // 2],
                                       stddev=Con.CONV2D_STDDEV), name=name)
            conv = self._tf.nn.conv2d(mid_conv, filter_w, strides=[1, 1, 1, 1], padding="SAME")

        elif conv_style["conv"] == 1:
            filter_w = self._tf.Variable(
                self._tf.random_normal([Con.FILTER_SIZE, Con.FILTER_SIZE, mid_conv.get_shape().as_list()[3], dim // 2],
                                       stddev=Con.CONV2D_STDDEV), name=name)
            conv = self._tf.nn.atrous_conv2d(mid_conv, filter_w, rate=conv_style["param"], padding="SAME")

        elif conv_style["conv"] == 2:
            filter_w_1 = self._tf.Variable(
                self._tf.random_normal(
                    [Con.FILTER_ASYMMETRIC, 1, mid_conv.get_shape().as_list()[3], mid_conv.get_shape().as_list()[3]],
                    stddev=Con.CONV2D_STDDEV), name=name)
            conv = self._tf.nn.conv2d(mid_conv, filter_w_1, strides=[1, 1, 1, 1], padding="SAME")
            filter_w_2 = self._tf.Variable(
                self._tf.random_normal(
                    [1, Con.FILTER_ASYMMETRIC, mid_conv.get_shape().as_list()[3], dim // 2],
                    stddev=Con.CONV2D_STDDEV), name=name)
            conv = self._tf.nn.conv2d(conv, filter_w_2, strides=[1, 1, 1, 1], padding="SAME")

        conv = self._tf.layers.batch_normalization(conv, center=True, scale=True, training=bn)
        conv = self._tf.nn.relu(conv)

        ## expand dim
        filter_1by1_expend = self._tf.Variable(
            self._tf.random_normal(shape=[1, 1, dim // 2, dim],
                                   stddev=Con.CONV2D_STDDEV,
                                   name=name_1by1_expend))
        conv = self._tf.nn.conv2d(conv, filter_1by1_expend, strides=[1, 1, 1, 1], padding="SAME")
        conv = self._tf.layers.batch_normalization(conv, center=True, scale=True, training=bn)
        conv = self._tf.nn.relu(conv)

        before_pooling = conv
        if is_pooling:
            conv = self._tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            conv = self._tf.concat([pooling, conv], axis=3)

        print("[{}] : {}".format(name, conv.get_shape().as_list()))
        print("[{}_prev] : {}".format(name, before_pooling.get_shape().as_list()))

        return conv, before_pooling

    def de_conv2d_bottleneck(self, name, input, add_channels, name_1by1_reduce, name_1by1_expend, dim_1by1, bn=True):
        ## conv part
        ### reduce dim
        filter_1by1_reduce = self._tf.Variable(
            self._tf.random_normal(shape=[1, 1, input.get_shape().as_list()[3], dim_1by1], stddev=Con.DE_CONV2D_STDDEV,
                                   name=name_1by1_reduce))
        mid_conv = self._tf.nn.conv2d(input, filter_1by1_reduce, strides=[1, 1, 1, 1], padding="SAME")
        mid_conv = self._tf.layers.batch_normalization(mid_conv, center=True, scale=True, training=bn)
        mid_conv = self._tf.nn.relu(mid_conv)

        ### de convolution.
        filter_de_w = self._tf.Variable(
            self._tf.random_normal([Con.DE_FILTER_SIZE, Con.DE_FILTER_SIZE, mid_conv.get_shape().as_list()[3] // 2,
                                    input.get_shape().as_list()[3]], stddev=Con.DE_CONV2D_STDDEV, name=name))
        de_conv = self._tf.nn.conv2d_transpose(input, filter_de_w,
                                               output_shape=[mid_conv.get_shape().as_list()[0],
                                                             mid_conv.get_shape().as_list()[1] * 2,
                                                             mid_conv.get_shape().as_list()[2] * 2,
                                                             mid_conv.get_shape().as_list()[3] // 2],
                                               strides=[1, 2, 2, 1], padding="SAME")
        de_conv = self._tf.layers.batch_normalization(de_conv, center=True, scale=True, training=bn)
        de_conv = self._tf.nn.relu(de_conv)

        ###expend dim
        filter_1by1_expend = self._tf.Variable(
            self._tf.random_normal(
                shape=[1, 1, mid_conv.get_shape().as_list()[3] // 2, input.get_shape().as_list()[3] // 2],
                stddev=Con.DE_CONV2D_STDDEV,
                name=name_1by1_expend))
        de_conv = self._tf.nn.conv2d(de_conv, filter_1by1_expend, strides=[1, 1, 1, 1], padding="SAME")
        de_conv = self._tf.layers.batch_normalization(de_conv, center=True, scale=True, training=bn)
        de_conv = self._tf.nn.relu(de_conv)

        ## add channels part
        if add_channels != None:
            de_conv = self._tf.concat([de_conv, add_channels], axis=3)

        print("[{}] : {}".format(name, de_conv.get_shape().as_list()))

        return de_conv

    def final_layer(self, name, input):
        pooling = self._tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        filter_w = self._tf.Variable(
            self._tf.random_normal([Con.FILTER_SIZE, Con.FILTER_SIZE, input.get_shape().as_list()[3], 1],
                                   stddev=Con.CONV2D_STDDEV), name=name)
        conv = self._tf.nn.conv2d(input, filter_w, strides=[1, 1, 1, 1], padding="SAME")

        # conv = self._tf.nn.sigmoid(conv)

        print("[{}] : {}".format(name, conv.get_shape().as_list()))
        return conv
