# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-03-10 19:05:23
# @Last Modified by:   massimo
# @Last Modified time: 2016-03-10 23:29:57
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from data_loader import load_mnist
import time

class MLP:
	def __init__(
		self,
		n_inputs=784, 
		n_classes=10, 
		layers=None, 
		activation='relu',
		optim='sgd', 
		dropout_p_input=0.0, 
		dropout_p_hidden=0.0, 
		learning_rate=0.01, 
		momentum=0.5, 
		batch_size=128,  
		numpy_rng=None, 
		theano_rng=None):

		self.n_inputs = n_inputs
		self.n_classes = n_classes
		self.dropout_p_input = dropout_p_input
		self.dropout_p_hidden = dropout_p_hidden
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.batch_size = batch_size
		if layers is None:
			layers = [256, 128]
		self.layers = layers

		optims = {
			'sgd': self.sgd
		}

		activs = {
			'tanh': self.tanh,
			'sigmoid': self.sigmoid,
			'relu': self.relu
		}
		assert optim in optims, 'Unknown optimization "{}"'.format(optim)
		self.optimization = optim
		self.optimization_fn = optims[optim]
		assert activation in activs, 'Unknown activation "{}"'.format(activation)
		self.activation = activation
		self.activation_fn = activs[self.activation]

		if numpy_rng is None:
			numpy_rng = np.random.RandomState(2**30)
		self.numpy_rng = numpy_rng
		if theano_rng is None:
			theano_rng = RandomStreams(2**30)
		self.theano_rng = theano_rng

		self.params, self.train_fn, self.pred_fn = None, None, None

	def init(self):
		# TODO: define MLP weigths as Theano SharedVariables
		# define the shared parameters
		self.W, self.b = [], []
		for i in range(len(self.layers)):
			# APPEND A NEW WEIGHT MATRIX OF HIDDEN LAYER
			# APPEND A NEW BIAS VECTOR OF THE HIDDEN LAYER
		self.Wy = # INITIALIZE THE HIDDEN-TO-OUTPUT WEIGHT MATRIX (like in LogReg)
		self.by = # INITIALIZE THE HIDDEN-TO-OUTPUT BIAS VECTOR (like in LogReg)
		params = self.W + self.b + [self.Wy, self.by]
		return params

	def model(self, X, dropout_p_input=0.0, dropout_p_hidden=0.0):
		# TODO: Define the MLP with the required number of hidden layers
		# dropout should be applyed at the input and after each hidden layer
		# use the self.activation_fn as activation

	def softmax(self, x):
		# numerically stable version of softmax (it avoids blowups due to high values of x)
		exp_x = T.exp(x - T.max(x, axis=1, keepdims=True))
		return exp_x / T.sum(exp_x, axis=1, keepdims=True)

	# ACTIVATION FUNCTIONS #
	def tanh(self, x):
		return T.tanh(x)

	def sigmoid(self, x):
		return T.nnet.sigmoid(x)

	def relu(self, x):
		#return T.max(0., x)
		return x * (x > 0)

	# DROPOUT #
	def dropout(self, x, p=0):
		# p (float): dropout probability on x
		# TODO: APPLY DROPOUT ON x only if p > 0
		return x

	# LOSS FUNCTION #
	def categorical_cross_entropy(self, y, y_hat):
		return T.mean(-T.log(y_hat[T.arange(y.shape[0]), y]), axis=0)

	def floatX(self, arr):
		return np.asarray(arr, dtype=theano.config.floatX)

	def init_weights(self, shape, sigma=0.01, name=''):
		if sigma == 0:
			W_bound = np.sqrt(6. / (shape[0] + shape[1]))
			return theano.shared(self.floatX(self.numpy_rng.uniform(low=-W_bound, high=W_bound, size=shape)), borrow=True, name=name)
		return theano.shared(self.floatX(self.numpy_rng.randn(*shape) * sigma), borrow=True, name=name)

	def sgd(self, cost, params, learning_rate=0.1):
		# compute the gradients of each parameter w.r.t. the loss
		pgrads = T.grad(cost, wrt=params)
		# define the sgd updates
		updates = OrderedDict([(p, p - learning_rate * g) for p, g in zip(params, pgrads)])
		return updates

	def apply_momentum(self, updates, momentum=0.5):
		# updates (dict or OrderedDict): reports for each parameter its symbolic update expression
		# TODO: apply momentum on the updates 
		return updates

	def fit(self, x, y, epochs=10, shuffle_training=True):
		if self.train_fn is None:
			print 'Compiling the training functions'
			# symbolic input and output variables
			x_sym, y_sym = T.matrix(), T.ivector()
			# build the model and get the output variable
			self.params = self.init()
			y_hat = self.model(x_sym, self.dropout_p_input, self.dropout_p_hidden)
			cost = self.categorical_cross_entropy(y_sym, y_hat)
			updates = self.optimization_fn(cost, self.params, self.learning_rate)
			if self.momentum > 0.:
				updates = self.apply_momentum(updates, self.momentum)
			self.train_fn = theano.function(inputs=[x_sym, y_sym], outputs=cost, updates=updates)

		if shuffle_training:
			shuffle_idx = self.numpy_rng.permutation(x.shape[0])
			x = x[shuffle_idx]
			y = y[shuffle_idx]
		num_train_batches = -(-x.shape[0] // self.batch_size)
		train_cost_history = []
		print 'Training started'
		for e in range(epochs):
			avg_cost = 0
			for bidx in range(num_train_batches):
				batch_x = x[bidx * self.batch_size: (bidx + 1) * self.batch_size]
				batch_y = y[bidx * self.batch_size: (bidx + 1) * self.batch_size]
				batch_cost = self.train_fn(batch_x, batch_y)
				train_cost_history.append(batch_cost)
				if np.isnan(batch_cost):
					print 'NaN cost detected. Abort'
					return
				avg_cost += batch_cost
			avg_cost /= num_train_batches
			print 'Epoch: {} Loss: {:.8f}'.format(e + 1, avg_cost)
		return train_cost_history

	def predict(self, x):
		if self.pred_fn is None:
			# build the prediction function
			x_sym = T.matrix()
			# disable any dropout in prediction
			y_hat_pred = self.model(x_sym, 0.0, 0.0)
			# then compute the predicted output as the class with maximum probability
			y_pred = T.argmax(y_hat_pred, axis=1)
			self.pred_fn = theano.function(inputs=[x_sym], outputs=y_pred)

		preds = np.asarray([])
		num_batches = -(-x.shape[0] // self.batch_size)
		for bidx in range(num_batches):
			batch_x = x[bidx * self.batch_size: (bidx + 1) * self.batch_size]
			batch_y_pred = self.pred_fn(batch_x)
			preds = np.concatenate((preds, batch_y_pred))
		return preds

	def set_params(self, learning_rate=0.1, momentum=0.5, dropout_p_input=0.0, dropout_p_hidden=0.0):
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.dropout_p_input = dropout_p_input
		self.dropout_p_hidden = dropout_p_hidden
		self.train_fn, self.pred_fn = None, None


if __name__ == '__main__':

	dataset = load_mnist('../../data/mnist.pkl.gz')
	train_x, train_y = dataset[0]
	valid_x, valid_y = dataset[1]
	test_x, test_y = dataset[2]

	# Try with different combinations of the parameters
	model = MLP(
		n_classes=10,
		n_inputs=train_x.shape[1],
		optim='rmsprop',
		activation='relu',
		dropout_p_input=0.0, 
		dropout_p_hidden=0.5, 
		layers=[256, 128], 
		learning_rate=0.001,
		momentum=0.0)

	t0 = time.time()
	model.fit(train_x, train_y, epochs=25)
	print 'Training completed in {:.2f} sec'.format(time.time() - t0)

	valid_y_pred = model.predict(valid_x)
	valid_accuracy = np.sum(valid_y_pred == valid_y, dtype=np.float32) / valid_y.shape[0]
	print 'Validation accuracy: {:.2f}'.format(valid_accuracy * 100)	# you should get around 97.5%

	test_y_pred = model.predict(test_x)
	test_accuracy = np.sum(test_y_pred == test_y, dtype=np.float32) / test_y.shape[0]
	print 'Test accuracy: {:.2f}'.format(test_accuracy * 100)	# you should get around 97.5%
