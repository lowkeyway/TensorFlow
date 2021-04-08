# -*- coding: utf-8 -*-
# @Time    : 2021/4/8 13:39
# @Author  : lowkeyway
import sys, os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def graphTest():
    b=tf.constant(value=0)
    c=tf.constant(value=1)
    print(c.graph is tf.get_default_graph())
    print("b.graph:", b.graph)
    print("c.graph:", c.graph)
    print("default graph:", tf.get_default_graph())

    g=tf.Graph()
    print("g:" ,g)
    with g.as_default():
        d=tf.constant(value=2, name="d")
        print("d:", d)
        print("d.graph:", d.graph)

    g2=tf.Graph()
    print("g2:", g2)
    e=tf.constant(value=3)
    print("e.graph:", e.graph)
    g2.as_default()
    f=tf.constant(value=4)
    print("f.graph:", f.graph)


def sessionTest():
    x=tf.constant(name="X", value=2)
    y=tf.constant(name="Y", value=3)
    z = x * y

    s1 = tf.Session()
    print("x:", x)
    print("y:", y)
    print("z:", z)
    print("s1:", s1)


    print("x:", s1.run(x))
    print("z:", s1.run(z))
    s1.close()

    with tf.Session() as s2:
        print("s2:", s2)
        print("z:", s2.run(z))

def main_func(argv):
    #graphTest()
    sessionTest()


if __name__ == '__main__':
    main_func(sys.argv)