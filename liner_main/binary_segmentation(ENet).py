import numpy as np
import tensorflow as tf
from network import MultiNet
from DataManager import data_manager
import config as Con
import cv2
import os
import matplotlib.pyplot as plt
import sys

# gpu setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# load images
images, labels = data_manager().get_all_images()
labels = np.expand_dims(labels, axis=3)

images = images / 255

# create network.
net = MultiNet(input=tf.placeholder(tf.float32, shape=[Con.BATCH_SIZE, 360, 640, 3], name="input_data"),
               label=tf.placeholder(tf.float32, shape=[Con.BATCH_SIZE, 360, 640, 1], name="label_data"),
               tensorflow=tf)

# net structure.
net.create_network(rate=Con.LEARNING_RATE)

with tf.Session() as sess:
    print("session_start")

    net.set_session(sess)
    sess.run(tf.global_variables_initializer())

    for i in range(Con.TOTAL_EPOCH):
        print("=================[epoch start : {}]=================".format(i + 1))

        for j in range(len(images) // Con.BATCH_SIZE):
            net.train_network(np.reshape(images[j:j + Con.BATCH_SIZE], [Con.BATCH_SIZE, 360, 640, 3]),
                              np.reshape(labels[j:j + Con.BATCH_SIZE], [Con.BATCH_SIZE, 360, 640, 1]))
            # print("\r[batch{}] :".format(j + 1))

            loss_val = net.calculate_binary_loss(np.reshape(images[j:j + Con.BATCH_SIZE], [Con.BATCH_SIZE, 360, 640, 3]),
                                                 np.reshape(labels[j:j + Con.BATCH_SIZE], [Con.BATCH_SIZE, 360, 640, 1]))
            sys.stdout.write("\r[batch{} loss] :{}".format(j + 1, loss_val))
            if j == len(images) // Con.BATCH_SIZE - 1:
                sys.stdout.write('\n')

        print("=================[epoch   end : {}]=================".format(i + 1))

        predict = net.calculate_prediction(np.reshape(images[0:2], [2, 360, 640, 3]))
        predict = np.reshape(predict, [2, 360, 640])
        plt.imshow(predict[0], cmap='gray', vmin=0, vmax=255)
        plt.show()
