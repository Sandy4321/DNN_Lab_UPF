# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-03-10 16:37:52
# @Last Modified by:   massimo
# @Last Modified time: 2016-03-10 23:29:57
import numpy as np
import theano
import theano.tensor as T

from collections import OrderedDict
from data_loader import load_mnist
import time

class LogisticRegression:
	def __init__(
		self,
		n_inputs=784, 
		n_classes=10, 
		learning_rate=0.05, 
		batch_size=128, 
		numpy_rng=None):
		self.n_inputs = n_inputs
		self.n_classes = n_classes
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		if numpy_rng is None:
			numpy_rng = np.random.RandomState(2**30)
		self.numpy_rng = numpy_rng

		self.params, self.train_fn, self.pred_fn = None, None, None

	def init(self):
		# TODO: define Logistic Regression weigths as Theano SharedVariables
		self.W = # define the weigth matrix
		self.b = # define the bias vector
		params = [self.W, self.b]
		return params 

	def model(self, X):
		# TODO: define the symbolic instuctions to compute the output probability function
		# of the Logistic Regression model
		pass

	def categorical_cross_entropy(self, y, y_hat):
		# y (int array, shape (self.batch_size,) ): actual output class 
		# y_hat (float32 array, shape (self.batch_size, self.n_classes)): expected class probability 
		# TODO: return the average cross-entropy the minibatch
		pass

	def floatX(self, arr):
		return np.asarray(arr, dtype=theano.config.floatX)

	def init_weights(self, shape, sigma=0.01, name=''):
		if sigma == 0:
			W_bound = np.sqrt(6. / (shape[0] + shape[1]))
			return theano.shared(self.floatX(self.numpy_rng.uniform(low=-W_bound, high=W_bound, size=shape)), borrow=True, name=name)
		return theano.shared(self.floatX(self.numpy_rng.randn(*shape) * sigma), borrow=True, name=name)

	def softmax(self, x):
		# x: (float32 array, shape (self.batch_size, self.n_classes))
		# TODO: compute the softmax of each row in x
		pass

	def sgd(self, cost, params, learning_rate):
		# TODO: Write the sgd updates of cost w.r.t. params, given the learning rate
		pass

	def fit(self, x, y, epochs=10, shuffle_training=True):
		if self.train_fn is None:
			print 'Compiling the training functions'
			X = T.matrix()
			y_sym = T.ivector()
			# initialize network's parameters
			self.params = self.init()
			y_hat = self.model(X)
			# symbolic operations to compute the categorical cross-entropy loss
			cost = self.categorical_cross_entropy(y_sym, y_hat)
			updates = self.sgd(cost, self.params, self.learning_rate)
			self.train_fn = theano.function(inputs=[X, y_sym], outputs=cost, updates=updates)

		if shuffle_training:
			shuffle_idx = self.numpy_rng.permutation(x.shape[0])
			x = x[shuffle_idx]
			y = y[shuffle_idx]
		num_train_batches = -(-x.shape[0] // self.batch_size)
		for e in range(epochs):
			avg_cost = 0
			for bidx in range(num_train_batches):
				batch_x = x[bidx * self.batch_size: (bidx + 1) * self.batch_size]
				batch_y = y[bidx * self.batch_size: (bidx + 1) * self.batch_size]
				batch_cost = self.train_fn(batch_x, batch_y)
				if np.isnan(batch_cost):
					print 'NaN cost detected. Abort'
					return
				avg_cost += batch_cost
			avg_cost /= num_train_batches
			print 'Epoch: {} Loss: {:.8f}'.format(e + 1, avg_cost)

	def predict(self, x):
		if self.pred_fn is None:
			X = T.matrix()
			y_hat = self.model(X)
			# symbolic operation to compute the predicted output of Logistic Regression
			y_pred = T.argmax(y_hat, axis=1)
			self.pred_fn = theano.function(inputs=[X], outputs=y_pred)
		preds = np.asarray([])
		num_batches = -(-x.shape[0] // self.batch_size)
		for bidx in range(num_batches):
			batch_x = x[bidx * self.batch_size: (bidx + 1) * self.batch_size]
			batch_y_pred = self.pred_fn(batch_x)
			preds = np.concatenate((preds, batch_y_pred))
		return preds

if __name__ == '__main__':
	dataset = load_mnist('../../data/mnist.pkl.gz')
	train_x, train_y = dataset[0]
	valid_x, valid_y = dataset[1]
	test_x, test_y = dataset[2]

	model = LogisticRegression(n_classes=10, n_inputs=train_x.shape[1], learning_rate=0.1)

	t0 = time.time()
	model.fit(train_x, train_y)
	print 'Training completed in {:.2f} sec'.format(time.time() - t0)

	# compute the validation accuracy (you should get values around 92%)
	valid_y_pred = model.predict(valid_x)
	valid_accuracy = np.sum(valid_y_pred == valid_y, dtype=np.float32) / valid_y.shape[0]
	print 'Validation accuracy: {:.2f}'.format(valid_accuracy * 100)

	# compute the test accuracy (you should get values around 92%)
	test_y_pred = model.predict(test_x)
	test_accuracy = np.sum(test_y_pred == test_y, dtype=np.float32) / test_y.shape[0]
	print 'Test accuracy: {:.2f}'.format(test_accuracy * 100)

