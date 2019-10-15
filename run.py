from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import logging
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("num", type=int,)
args = parser.parse_args()

NUM = args.num
def main():
    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    wl, xl = [], [x]
    for k in range(NUM):
        wl.append(tf.Variable(tf.zeros([784, 784])))
        xl.append(tf.matmul(xl[-1], wl[-1]))

    y = tf.nn.softmax(tf.matmul(xl[-1], W) + b)

    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(
        0.01).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(10000000):
        if i % 100 == 0:
            print("step{}".format(i))
        batch_xs, batch_ys = mnist.train.next_batch(100)
        options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        sess.run(train_step, feed_dict={
            x: batch_xs, y_: batch_ys}, options=options)
    print("train finish")

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(sess.run(accuracy, feed_dict={
          x: mnist.test.images, y_: mnist.test.labels}))

main()
