# -*- coding: utf-8 -*-
# @Time    : 2021/4/21 17:59
# @Author  : lowkeyway
import sys,os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def constantTest():
    tf.enable_eager_execution()
    rank_0_tensor = tf.constant(4)
    print(rank_0_tensor)
    rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
    print(rank_1_tensor)
    rank_2_tensor = tf.constant([[1, 2],
                                 [3, 4],
                                 [5, 6]], dtype=tf.float16)
    print(rank_2_tensor)

    c1 = tf.constant('Hello, TensorFlow')
    print("c1: ", c1)

    zero_t = tf.zeros([2, 3], tf.int32)
    print("zero_t: ", zero_t)

    ones_t = tf.ones([2, 3], tf.int32)
    print("ones_t: ", ones_t)

    linspace = tf.linspace(2.0, 5.0, 5)
    print("linspace: ", linspace)

    range_t = tf.range(10)
    print("range_t: ", range_t)

def variableTest():
    tf.enable_eager_execution()
    v = tf.Variable(10)
    v1 = tf.Variable(tf.random_normal([4, 4], mean=0, stddev=4, dtype=tf.float32), name="v1")
    v2 = tf.Variable(tf.random_uniform(shape=[16], minval=0.0, maxval=255.0, dtype=tf.float32), name="v2")
    v3 = tf.Variable(v2.initialized_value(), name="v3")

    print("v = \n", v)
    print("v1 = \n", v1)
    print("v1 = \n", v1.numpy())
    print("v2 = \n", v2.numpy())
    print("v3 = \n", v3.numpy())

def placeTest():
    x = tf.placeholder("float")
    y = 2 * x
    data = tf.random_uniform([4, 5], 10)
    with tf.Session() as sess:
        xData = sess.run(data)
        print(sess.run(y, feed_dict = {x : xData}))


def main_func(argv):

    #constantTest()
    #variableTest()
    #placeTest()
    get_variable_demo()

if __name__ == '__main__':
    main_func(sys.argv)