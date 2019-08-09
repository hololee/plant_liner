from layers import layers as ll


class MultiNet:

    def __init__(self, input, label, tensorflow):
        """
        :param input:
        :param label:
        :param sess:
        """
        # input place holder.
        self._input_placeholder = input
        self._label_placeholder = label
        self._bn_flag = tensorflow.placeholder(tensorflow.bool)
        self._tf = tensorflow

    def create_network(self, rate=0.1):
        """
        :return:
        """

        layers = ll(tf=self._tf)
        ### layer1

        layer = layers.initial("init", self._input_placeholder, bn=self._bn_flag)
        layer, _ = layers.conv2d("conv0", layer, 48, "reduce_0", "expend_0", 50,
                                 is_pooling=True, bn=self._bn_flag)
        layer, _ = layers.conv2d("conv1", layer, 64, "reduce_1", "expend_1", 50,
                                           bn=self._bn_flag)
        ### layer3
        layer, layer2_prev = layers.conv2d("down2", layer, 64, "reduce_6", "expend_6", 90,
                                           is_pooling=True, bn=self._bn_flag)

        layer, _ = layers.conv2d("conv4", layer, 128, "reduce_7", "expend_7", 90,
                                 bn=self._bn_flag)

        layer, _ = layers.conv2d("conv5", layer, 128, "reduce_8", "expend_8", 90,
                                 conv_style={"conv": layers.CONV_DILATED, "param": 2}, bn=self._bn_flag)

        layer, _ = layers.conv2d("conv6", layer, 128, "reduce_9", "expend_9", 90,
                                 conv_style={"conv": layers.CONV_ASYMMETRIC, "param": 0}, bn=self._bn_flag)

        layer, _ = layers.conv2d("conv7", layer,128, "reduce_10", "expend_10", 90,
                                 conv_style={"conv": layers.CONV_DILATED, "param": 4}, bn=self._bn_flag)

        layer, _ = layers.conv2d("conv8", layer, 128, "reduce_11", "expend_11", 90,
                                 bn=self._bn_flag)

        layer, _ = layers.conv2d("conv9", layer, 128, "reduce_12", "expend_12", 90,
                                 conv_style={"conv": layers.CONV_DILATED, "param": 8}, bn=self._bn_flag)

        layer, _ = layers.conv2d("conv10", layer, 128, "reduce_13", "expend_13", 90,
                                 conv_style={"conv": layers.CONV_ASYMMETRIC, "param": 0}, bn=self._bn_flag)

        layer, _ = layers.conv2d("conv11", layer, 128, "reduce_14", "expend_14", 90,
                                 conv_style={"conv": layers.CONV_DILATED, "param": 16}, bn=self._bn_flag)

        layer = layers.de_conv2d_bottleneck("up1", layer, None, "reduce_15", "expend_15", 50, bn=self._bn_flag)

        layer, _ = layers.conv2d("conv12", layer, layer.get_shape().as_list()[3], "reduce_15", "expend_15", 32,
                                 bn=self._bn_flag)

        layer, _ = layers.conv2d("conv13", layer, layer.get_shape().as_list()[3], "reduce_16", "expend_16", 32,
                                 bn=self._bn_flag)

        layer = layers.de_conv2d_bottleneck("up2", layer, None, "reduce_17", "expend_17", 25, bn=self._bn_flag)

        layer, _ = layers.conv2d("conv14", layer, layer.get_shape().as_list()[3], "reduce_18", "expend_18", 16,
                                 bn=self._bn_flag)

        layer = layers.de_conv2d_bottleneck("up3", layer, None, "reduce_19", "expend_19", 10, bn=self._bn_flag)

        layer = layers.final_layer("final", layer)

        self._binary_logit = layer
        # self._loss = self._tf.reduce_mean(self._tf.nn.sigmoid_cross_entropy_with_logits(labels=self._label_placeholder,
        #                                                                                 logits=self._binary_logit))

        self._loss = self._tf.reduce_mean(self._tf.nn.softmax_cross_entropy_with_logits(labels=self._label_placeholder,
                                                                                        logits=self._binary_logit))

        self._optimizer = self._tf.train.AdamOptimizer(learning_rate=rate).minimize(self._loss)
        # self._optimizer = self._tf.train.MomentumOptimizer(learning_rate=rate, momentum=0.9).minimize(self._loss)

    def set_session(self, sess):
        self._tf_session = sess

    def calculate_prediction(self, data):
        """
        :param data:
        :return:
        """
        return self._tf_session.run(self._binary_logit, feed_dict={self._input_placeholder: data,
                                                                   self._bn_flag: False})

    def calculate_binary_loss(self, data, labels):
        """
        :param data:
        :param labels:
        :return:
        """
        loss_val = self._tf_session.run(self._loss,
                                        feed_dict={self._input_placeholder: data, self._label_placeholder: labels,
                                                   self._bn_flag: False})
        return loss_val

    def train_network(self, data, labels):
        """
        :param data:
        :param labels:
        :param rate:
        :return:
        """
        extra_update_ops = self._tf.get_collection(self._tf.GraphKeys.UPDATE_OPS)
        _, _ = self._tf_session.run([self._optimizer, extra_update_ops],
                                    feed_dict={self._input_placeholder: data, self._label_placeholder: labels,
                                               self._bn_flag: True})
