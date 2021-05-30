# -*- coding: utf-8 -*-
# @Time    : 5/28/2021 3:05 PM
# @Author  : lowkeyway
"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os,sys
import tensorflow.python.platform
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
SOURCE_URL      = 'http://yann.lecun.com/exdb/mnist/'
VALIDATION_SIZE = 5000
TRAIN_IMAGES    = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS    = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES     = 't10k-images-idx3-ubyte.gz'
TEST_LABELS     = 't10k-labels-idx1-ubyte.gz'


def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath

def read_data_sets(train_dir):
    local_file = maybe_download(TRAIN_IMAGES, train_dir)
    local_file = maybe_download(TRAIN_LABELS, train_dir)
    local_file = maybe_download(TEST_IMAGES, train_dir)
    local_file = maybe_download(TEST_LABELS, train_dir)


def main_func(argv):
    read_data_sets(".")


if __name__ == '__main__':
    main_func(sys.argv)