# -*- coding: utf-8 -*-
# @Time    : 2021/2/3 22:05
# @Author  : lowkeyway
import sys
import tensorflow as tf
import os
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

def main_func(argv):
    print("TensorFlow Version: ", tf.__version__)
    print("Python Version: ", platform.python_version())
    tf.enable_eager_execution()
    hello = tf.constant("Hello, tensorflow!")
    print(hello.numpy())


if __name__ == '__main__':
    main_func(sys.argv)