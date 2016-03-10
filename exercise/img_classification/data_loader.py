# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-03-10 15:33:49
# @Last Modified by:   massimo
# @Last Modified time: 2016-03-10 23:29:56
import numpy as np
import gzip
import cPickle as pkl   # python 2.x

def load_gz(gz_path):
	dataset = pkl.load(gzip.open(gz_path, 'rb'))
	return dataset

def load_mnist(mnist_path='../data/mnist.pkl.gz'):
	train_data, valid_data, test_data = load_gz(mnist_path)
	train_x, train_y = train_data[0], train_data[1].astype(np.int32)
	valid_x, valid_y = valid_data[0], valid_data[1].astype(np.int32)
	test_x, test_y = test_data[0], test_data[1].astype(np.int32)
	return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

