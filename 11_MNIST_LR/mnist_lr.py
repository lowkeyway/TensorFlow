# -*- coding: utf-8 -*-
# @Time    : 5/30/2021 4:22 PM
# @Author  : lowkeyway
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
print(tf.__version__)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_size = 128
lr = 0.01


def line_regression_demo():
    input_x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="images")
    labels = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="labels")
    weights = tf.get_variable(shape=[784, 10], initializer=tf.random_normal_initializer(stddev=0.01), name="weight")
    bias = tf.get_variable(shape=[1, 10], initializer=tf.constant_initializer(0), name="bias")

    # 线性模型
    y = tf.add(tf.matmul(input_x, weights), bias)

    # 损失
    diff = tf.square(tf.subtract(y, labels))
    loss = tf.reduce_mean(diff)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    # 准确率
    correct = tf.equal(tf.argmax(labels, 1), tf.argmax(y, 1))
    acc = tf.reduce_sum(tf.cast(correct, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            train_x, train_y = mnist.train.next_batch(batch_size)
            _, loss_value = sess.run([optimizer, loss], feed_dict={input_x: train_x, labels: train_y})
            if i % 100 == 0:
                print("loss_value : ", loss_value)
        test_x, test_y = mnist.test.next_batch(1000)
        acc_ = sess.run(acc, feed_dict={input_x:test_x, labels:test_y})
        print("final accuracy : %.2f"%acc_)

def main_func(argv):
    line_regression_demo()


if __name__ == '__main__':
    main_func(sys.argv)