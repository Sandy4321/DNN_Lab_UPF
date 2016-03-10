# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-03-10 15:33:49
# @Last Modified by:   massimo
# @Last Modified time: 2016-03-10 23:29:56
import theano
import theano.tensor as T
import numpy as np

def floatX(arr):
	return np.asarray(arr, dtype=theano.config.floatX)

# ACTIVATION FUNCTIONS #
def tanh(x):
	return T.tanh(x)

def sigmoid(x):
	return T.nnet.sigmoid(x)

def relu(x):
	return x * (x > 0)

# WEIGHT INITIALIZATION FUNCTIONS
def init_weights(shape, sigma=0.01, name='', rng=None):
	if rng is None:
		rng = np.random.RandomState(2**30)
	if sigma == 0:
		W_bound = np.sqrt(6. / (shape[0] + shape[1]))
		return theano.shared(floatX(rng.uniform(low=-W_bound, high=W_bound, size=shape)), borrow=True, name=name)
	return theano.shared(floatX(rng.randn(*shape) * sigma), borrow=True, name=name)

def init_ortho(dim, num_matrices=1, stack_axis=1, name='', rng=None):
	if rng is None:
		rng = np.random.RandomState(2**30)
	U = []
	for i in range(num_matrices):
		W = rng.randn(dim, dim)
		U.append(np.linalg.svd(W)[0])
	return theano.shared(floatX(np.concatenate(U, axis=stack_axis)), borrow=True, name=name)

def init_eye(shape, name='', rng=None):
	if rng is None:
		rng = np.random.RandomState(2**30)
	if shape[0] != shape[1]:
		return init_weights(shape, 0, name, rng)
	else:
		return theano.shared(floatX(np.eye(*shape, dtype=theano.config.floatX), borrow=True, name=name))


