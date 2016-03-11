# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-03-08 10:36:46
# @Last Modified by:   massimo
# @Last Modified time: 2016-03-11 15:54:43
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
	# numerically stable version of softmax (it avoids blowups due to high values of x)
	exp_x = T.exp(x - T.max(x, axis=1, keepdims=True))
	return exp_x / T.sum(exp_x, axis=1, keepdims=True)

train_data, valid_data, test_data = load_gz('../../data/mnist.pkl.gz')
train_x, train_y = train_data[0], train_data[1].astype(np.int32)
valid_x, valid_y = valid_data[0], valid_data[1].astype(np.int32)
test_x, test_y = test_data[0], test_data[1].astype(np.int32)

n_inputs = train_x.shape[1]
n_classes = 10
learning_rate = 0.1
batch_size = 128
epochs = 10

num_train_batches = -(-train_x.shape[0] // batch_size)
num_valid_batches = -(-valid_x.shape[0] // batch_size)
num_test_batches = -(-test_x.shape[0] // batch_size)

np.random.seed(2**30)
# define the input variables as Theano Tensors
X = T.matrix()
y = T.ivector()

# define weigths as Theano SharedVariables
W = init_weights((n_inputs, n_classes))
b = theano.shared(value=floatX(np.zeros((n_classes,))), borrow=True)
params = [W, b]

# symbolic operation to compute the predicted output of Logistic Regression
y_hat = softmax(T.dot(X, W) + b)
y_pred = T.argmax(y_hat, axis=1)
# categorical cross-entropy loss
cost = T.mean(-T.log(y_hat[T.arange(y.shape[0]), y]))

# compute the gradients of each parameter w.r.t. the cross-entropy loss
pgrads = T.grad(cost, wrt=params)
# define the sgd updates
updates = OrderedDict([(p, p - learning_rate * g) for p, g in zip(params, pgrads)])

# then define the training and prediction functions
loss_fn = theano.function(inputs=[X, y], outputs=cost)
train_fn = theano.function(inputs=[X, y], outputs=cost, updates=updates)
pred_fn = theano.function(inputs=[X], outputs=y_pred)

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


