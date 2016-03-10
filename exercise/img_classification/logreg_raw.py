# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-03-10 17:09:49
# @Last Modified by:   massimo
# @Last Modified time: 2016-03-10 23:29:57
import numpy as np
import theano
import theano.tensor as T

from collections import OrderedDict
import gzip
import cPickle as pkl   # python 2.x
import time

def load_gz(gz_path):
	dataset = pkl.load(gzip.open(gz_path, 'rb'))
	return dataset

def floatX(arr):
	return np.asarray(arr, dtype=theano.config.floatX)

def init_weights(shape, sigma=0.01, name=''):
	if sigma == 0:
		W_bound = np.sqrt(6. / (shape[0] + shape[1]))
		return theano.shared(floatX(np.random.uniform(low=-W_bound, high=W_bound, size=shape)), borrow=True, name=name)
	return theano.shared(floatX(np.random.randn(*shape) * sigma), borrow=True, name=name)

def softmax(x):
	# TODO: WRITE YOUR DEFINITION OF SOFTMAX HERE
	pass

# import the MNIST dataset
# it has been alredy split into train, validation and test sets
train_data, valid_data, test_data = load_gz('../data/mnist.pkl.gz')
train_x, train_y = train_data[0], train_data[1].astype(np.int32)
valid_x, valid_y = valid_data[0], valid_data[1].astype(np.int32)
test_x, test_y = test_data[0], test_data[1].astype(np.int32)

# define our model parameters here
n_inputs = train_x.shape[1]			# number of input features
n_classes = 10						# number of output classes (it depends on the task, 10 for MNIST)
learning_rate = 0.1 				# learning rate used in Stochastic Gradient Descent
batch_size = 128					# number of samples per minibatch
epochs = 10							# number of training epochs (i.e. number of passes over the entire training set)

# compute the number of minibateches
num_train_batches = -(-train_x.shape[0] // batch_size)
num_valid_batches = -(-valid_x.shape[0] // batch_size)
num_test_batches = -(-test_x.shape[0] // batch_size)

np.random.seed(2**30)

####################################################################
# TODO: write the Theano code for the Logistic Regression classifier
# You have to do the following:
# 1) define the input variables as Theano Tensors
# 2) define weights as Theano SharedVariables
# 3) define the symbolic operation to compute the predicted class probability
# 	the predicted output class of Logistic Regression
# 4) define the categorical cross-entropy loss
# 5) compute the gradients of each parameter w.r.t. the cross-entropy loss
# 6) define the sgd updates
# 7) finally, define the loss, training and prediction functions (call them loss_fn, train_fn and pred_fn)


# PUT YOUR CODE HERE

############################################################

# This is a simple check for correctness of the entire model
# With an untrained model, predictions should be almost uniformly distributed over each class
# Thus the expected loss is -log(1/n_classes) = log(n_classes) ~ 2.3
# This simple check can reveal errors in your loss function or in other parts of your model (e.g, the softmax normalization,...)
expected = np.log(len(np.unique(train_y)))
actual = loss_fn(train_x, train_y)
print 'Expected initial loss: ', expected
print 'Actual initial loss: ', actual

# randomly shuffle the training data
shuffle_idx = np.random.permutation(train_x.shape[0])
train_x = train_x[shuffle_idx]
train_y = train_y[shuffle_idx]

print 'Training started'
t0 = time.time()
for e in range(epochs):
	avg_cost = 0
	for bidx in range(num_train_batches):
		batch_x = train_x[bidx * batch_size: (bidx + 1) * batch_size]
		batch_y = train_y[bidx * batch_size: (bidx + 1) * batch_size]
		batch_cost = train_fn(batch_x, batch_y)
		avg_cost += batch_cost
	avg_cost /= num_train_batches
	print 'Epoch: {} Loss: {:.8f}'.format(e + 1, avg_cost)
print 'Training completed in {:.2f} sec'.format(time.time() - t0)


# compute the validation accuracy (you should get values around 92%)
hits = 0
for bidx in range(num_valid_batches):
	batch_x = valid_x[bidx * batch_size: (bidx + 1) * batch_size]
	batch_y = valid_y[bidx * batch_size: (bidx + 1) * batch_size]
	batch_y_pred = pred_fn(batch_x)
	hits += np.sum(batch_y_pred == batch_y)
accuracy = np.float32(hits) / valid_y.shape[0]
print 'Valid. accuracy: {:.4f}'.format(accuracy)


# compute the test accuracy (you should get values around 92%)
hits = 0
for bidx in range(num_test_batches):
	batch_x = test_x[bidx * batch_size: (bidx + 1) * batch_size]
	batch_y = test_y[bidx * batch_size: (bidx + 1) * batch_size]
	batch_y_pred = pred_fn(batch_x)
	hits += np.sum(batch_y_pred == batch_y)
accuracy = np.float32(hits) / test_y.shape[0]
print 'Test. accuracy: {:.4f}'.format(accuracy)


