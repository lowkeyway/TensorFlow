# -*- coding: utf-8 -*-
# @Time    : 2021/2/22 14:00
# @Author  : lowkeyway
import sys
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def addTest():
    two_node = tf.constant(2)
    three_node = tf.constant(3)
    sum_node = tf.add(two_node, three_node)
    print("two_node: ", two_node)
    print("three_node: ", three_node)
    print("sum_node: ", sum_node)
    sess = tf.Session()
    print(sess.run(sum_node))

def tensorTest():
    tf.enable_eager_execution()
    rank_0_tensor = tf.constant(4)
    print(rank_0_tensor)
    rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
    print(rank_1_tensor)
    rank_2_tensor = tf.constant([[1, 2],
                                 [3, 4],
                                 [5, 6]], dtype=tf.float16)
    print(rank_2_tensor)
    rank_3_tensor = tf.constant([
        [[0, 1, 2, 3, 4],
         [5, 6, 7, 8, 9]],
        [[10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19]],
        [[20, 21, 22, 23, 24],
         [25, 26, 27, 28, 29]], ])

    print(rank_3_tensor)

    c1 = tf.constant('Hello, TensorFlow')
    c2 = tf.constant('Hello, TensorFlow', dtype=tf.string, name="c1")
    print("c1: ", c1)
    print("c2: ", c2)

def main_func(argv):
    # tensorTest()
    addTest()

if __name__ == '__main__':
    main_func(sys.argv)