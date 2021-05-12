# -*- coding: utf-8 -*-
# @Time    : 2021/5/12 15:36
# @Author  : lowkeyway
import sys,os
import tensorflow as tf
import  numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def assign_test():
    # tf.enable_eager_execution()
    a = tf.Variable(0, name="counter")
    b = tf.add(a, tf.constant(1, name="bias"))
    c = tf.Variable([1, 2], name="table")
    d = tf.assign(c, [3, 4])
    print("a: ", a)
    print("b: ", b)

    update = tf.assign(a, b)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("c: ", sess.run(c))
        sess.run(d)
        print("c_assign: ", sess.run(c))
        for i in range(1000):
            _ = sess.run(update)
        a_value = sess.run(b)
        print("final counter : ", a_value)

def fetch_test():
    a = tf.Variable(0, name="counter")
    d1 = tf.placeholder(dtype=tf.float32, shape=[2, 2])
    d2 = tf.placeholder(dtype=tf.float32, shape=[2, 1])
    d3 = tf.matmul(d1, d2)


    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print("a: ", a)
    print("a_run: ", sess.run(a))

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[4.], [4.]])
    z = tf.matmul(x, y)
    print("z : ", z)
    print("z_run: ", sess.run(z, feed_dict={d1: x, d2: y}))

    sess.close()

def feed_test():
    input_x = tf.placeholder(dtype=tf.float32, shape=[4], name="x")
    input_y = tf.placeholder(dtype=tf.float32, shape=[4], name="y")
    a = tf.constant(3.0, dtype=tf.float32)
    b = np.array([1, 2, 3, 4])
    c = np.array([5, 6, 7, 8])
    # b = tf.Variable(tf.random_normal([3, 3], stddev=3.0), dtype=tf.float32)
    # c = tf.Variable(tf.random_normal([3, 3], stddev=3.0), dtype=tf.float32)

    print("input_x: ", input_x)
    print("input_y: ", input_y)

    sum = tf.add(input_x, a)
    mul = tf.multiply(input_x, input_y)

    init = tf.global_variables_initializer()
    sess = tf.Session(graph=tf.get_default_graph())
    # device placement
    with tf.Session() as sess:
        sess.run(init)
        curr_sum = sess.run(sum, feed_dict={input_x:[1, 2, 3, 4]})
        curr_mul = sess.run(mul, feed_dict={input_x:b, input_y:c})
        print("sum : ", curr_sum)
        print("mul : ", curr_mul)

def main_func(argv):
    # assign_test()
    # fetch_test()
    feed_test()

if __name__ == '__main__':
    main_func(sys.argv)