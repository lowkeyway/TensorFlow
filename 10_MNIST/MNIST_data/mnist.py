# -*- coding: utf-8 -*-
# @Time    : 5/26/2021 3:31 PM
# @Author  : lowkeyway


from __future__ import division
from __future__ import print_function
import os
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

file_list = [
    "train-images.idx3-ubyte",
    "train-labels.idx1-ubyte",
    "t10k-images.idx3-ubyte",
    "t10k-labels.idx1-ubyte",
]

class MNIST_IMAGE:
    MagicNum = None
    ImageNum = None
    ImageRow = None
    ImageCol = None
    ImageList = []

    def __init__(self, fileObj):
        print("MNIST_IMAGE Init")

        raw_header = fileObj.read(4)
        self.MagicNum, = struct.unpack(">I", raw_header)

        raw_header = fileObj.read(4)
        self.ImageNum, = struct.unpack(">I", raw_header)

        raw_header = fileObj.read(4)
        self.ImageRow, = struct.unpack(">I", raw_header)

        raw_header = fileObj.read(4)
        self.ImageCol, = struct.unpack(">I", raw_header)

        for index in range(self.ImageNum):
            img = fileObj.read(self.ImageRow * self.ImageCol)
            tp = struct.unpack(">784B", img)
            image = np.asarray(tp)
            image = image.reshape((28, 28))
            self.ImageList.append(image)


    def showHeader(self):
        print("MNIST_IMAGE header is : ", self.MagicNum, self.ImageNum, self.ImageRow, self.ImageCol)

    def showImage(self, Index = 0, fullSize = 0):
        print("Index = ", Index, ", fullSize = ", fullSize)
        if fullSize:
            return
        plt.imshow(self.ImageList[Index], cmap=plt.cm.gray)
        plt.show()

class MNIST_LABEL:
    MagicNum = None
    LabelNum = None
    LabelList = []

    def __init__(self, fileObj):
        print("MNIST_LABEL Init")

        raw_header = fileObj.read(4)
        self.MagicNum, = struct.unpack(">I", raw_header)

        raw_header = fileObj.read(4)
        self.LabelNum, = struct.unpack(">I", raw_header)

        for index in range(self.LabelNum):
            raw = fileObj.read(1)
            label, = struct.unpack(">B", raw)
            self.LabelList.append(label)

    def showHeader(self):
        print("MNIST_LABEL header is : ", self.MagicNum, self.LabelNum)

    def showLabel(self, Index = 0):
        print("Label[%d] = %d" %(Index, self.LabelList[Index]))


def create_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_file_full_name(path, name):
    create_path(path)
    if path[-1] == "/":
        full_name = path + name
    else:
        full_name = path + "/" + name
    return full_name


def read_mnist(file_name):
    file_path = "./data"
    full_path = get_file_full_name(file_path, file_name)
    file_object = open(full_path, 'rb')  # python3 need rb  python2 r is ok
    return file_object

def mnist_test():
    Index = 20
    images_file_name = file_list[0]
    labels_file_name = file_list[1]
    images_file = read_mnist(images_file_name)
    labels_file = read_mnist(labels_file_name)
    MI = MNIST_IMAGE(images_file)
    MI.showHeader()
    MI.showImage(Index)

    ML = MNIST_LABEL(labels_file)
    ML.showHeader()
    ML.showLabel(Index)

def main_func(argv):
    mnist_test()

if __name__ == '__main__':
    main_func(sys.argv)