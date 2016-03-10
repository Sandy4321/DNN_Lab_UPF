# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-03-10 22:09:18
# @Last Modified by:   massimo
# @Last Modified time: 2016-03-10 23:29:56
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet.conv import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from collections import OrderedDict

from data_loader import load_mnist
import time
import os

class ConvNet:
	def __init__(self,
				input_shape,
			    n_classes=10,
			    optim='adagrad',
			    activation='relu',
			    num_kernels=None,
			    hidden_layers=None,
			    batch_size=128,
			    dropout_p_input=0.0,
			    dropout_p_conv=0.0,
			    dropout_p_hidden=0.0,
			    learning_rate=0.01,
			    momentum=0.5,
			    numpy_rng=None,
			    theano_rng=None):
		self.input_shape = input_shape	# (num input feature maps, image height, image width)
		if num_kernels is None:		# number of kernels per layer
			num_kernels = [20, 50]
		if hidden_layers is None:	# number of fully connected hidden layers
			hidden_layers = [128]
		self.hidden_layers = hidden_layers
		self.num_kernels = num_kernels
		self.n_classes = n_classes
		self.dropout_p_input = dropout_p_input
		self.dropout_p_conv = dropout_p_conv
		self.dropout_p_hidden = dropout_p_hidden
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.batch_size = batch_size

		optims = {
			'sgd': self.sgd,
			'adagrad': self.adagrad,
			'rmsprop': self.rmsprop
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

		self.train_fn, self.pred_fn = None, None

	def get_conv_output_shape(self, input_shape, filter_shape, stride=1):
		return (
			input_shape[0],
			filter_shape[0],
			int((input_shape[2] - filter_shape[2] + 1) / stride),
			int((input_shape[3] - filter_shape[3] + 1) / stride)
			)

	def init(self):
		input_feat_maps, input_height, input_width = self.input_shape

		# ConvLayer1
		W1_input_shape = (self.batch_size, input_feat_maps, input_height, input_width)
		W1_filter_shape = (self.num_kernels[0], input_feat_maps, 5, 5)
		W1_output_shape = self.get_conv_output_shape(W1_input_shape, W1_filter_shape)
		W1_pool_shape = (2, 2)
		W1_pool_output_shape = (W1_output_shape[0], W1_output_shape[1], int(W1_output_shape[2] / W1_pool_shape[0]), int(W1_output_shape[3] / W1_pool_shape[1]))

		print 'Input shape:', self.input_shape
		print 'Conv1: filter shape', W1_filter_shape
		print 'Conv1: output shape', W1_output_shape
		print 'Pool1: shape', W1_pool_shape
		print 'Pool1: output shape', W1_pool_output_shape

		self.W1 = self.init_filter(W1_filter_shape, input_shape=W1_input_shape, output_shape=W1_pool_output_shape, name='W1')
		self.b1 = theano.shared(value=np.zeros((W1_filter_shape[0],), dtype=theano.config.floatX), borrow=True) # one bias per feature map (kernel)

		# ConvLayer2
		W2_input_shape = (W1_output_shape[0], W1_output_shape[1], int(W1_output_shape[2] / W1_pool_shape[0]), int(W1_output_shape[3] / W1_pool_shape[1]))
		W2_filter_shape = (self.num_kernels[1], W1_output_shape[1], 5, 5)
		W2_output_shape = self.get_conv_output_shape(W2_input_shape, W2_filter_shape)
		W2_pool_shape = (2, 2)
		W2_pool_output_shape = (W2_output_shape[0], W2_output_shape[1], int(W2_output_shape[2] / W2_pool_shape[0]), int(W2_output_shape[3] / W2_pool_shape[1]))
		
		print 'Conv2: filter shape', W2_filter_shape
		print 'Conv2: output shape', W2_output_shape
		print 'Pool2: shape', W2_pool_shape
		print 'Pool2: output shape', W2_pool_output_shape

		self.W2 = self.init_filter(W2_filter_shape, input_shape=W2_input_shape, output_shape=W2_pool_output_shape, name='W2')
		self.b2 = theano.shared(value=np.zeros((W2_filter_shape[0],), dtype=theano.config.floatX), borrow=True) # one bias per feature map (kernel)

		# Fully connected layers
		hid_n_inputs = np.prod(W2_pool_output_shape[1:])
		self.W, self.b = [], []
		for i in range(len(self.hidden_layers)):
			print 'Hidden{}: num. units {}'.format(i+1, self.hidden_layers[i])
			self.W.append(self.init_weights((hid_n_inputs if i == 0 else self.hidden_layers[i-1], self.hidden_layers[i]), sigma=0, name='W_{}'.format(i)))
			self.b.append(theano.shared(value=np.zeros((self.hidden_layers[i],), dtype=theano.config.floatX), borrow=True, name='b_{}'.format(i)))

		# Final Logistic Regression layer
		self.Wy = self.init_weights((self.hidden_layers[-1], self.n_classes), sigma=0, name='Wy')
		self.by = theano.shared(value=np.zeros((self.n_classes,), dtype=theano.config.floatX), borrow=True, name='by')

		params = [self.W1, self.W2, self.b1, self.b2, self.Wy, self.by] + self.W + self.b
		return params

	def model(self, X, dropout_p_input=0.0, dropout_p_conv=0.0, dropout_p_hidden=0.0):
		X = self.dropout(X, dropout_p_input)

		# NOTE: since there's one bias per feature map, the bias vector to should be reshaped
		# to be a 4d tensor of shape (1, n_filters, 1, 1) to be broadcasted correctly
		l_conv_1 = self.relu(conv2d(X, self.W1) + self.b1.dimshuffle('x', 0, 'x', 'x'))
		l_conv_1 = max_pool_2d(l_conv_1, (2, 2), ignore_border=True)
		l_conv_1 = self.dropout(l_conv_1, dropout_p_conv)

		l_conv_2 = self.relu(conv2d(l_conv_1, self.W2) + self.b2.dimshuffle('x', 0, 'x', 'x')) 
		l_conv_2 = max_pool_2d(l_conv_2, (2, 2), ignore_border=True)
		l_conv_2 = T.flatten(l_conv_2, outdim=2)
		l_conv_2 = self.dropout(l_conv_2, dropout_p_hidden)

		for i in range(len(self.hidden_layers)):
			l_hid = self.relu(T.dot(l_conv_2 if i == 0 else l_hid, self.W[i]) + self.b[i])
			l_hid = self.dropout(l_hid, dropout_p_hidden)
		y = self.softmax(T.dot(l_hid, self.Wy) + self.by)
		return y 

	def softmax(self, x):
		# numerically stable version of softmax (it avoids exp to blowup due to high values of x)
		exp_x = T.exp(x - T.max(x, axis=1, keepdims=True))
		return exp_x / T.sum(exp_x, axis=1, keepdims=True)

	# ACTIVATION FUNCTIONS #
	def tanh(self, x):
		return T.tanh(x)

	def sigmoid(self, x):
		return T.nnet.sigmoid(x)

	def relu(self, x):
		return x * (x > 0)

	# DROPOUT #
	def dropout(self, x, p=0):
		if p > 0:
			retain_p = 1.0 - p
			x *= self.theano_rng.binomial(x.shape, p=retain_p, dtype=theano.config.floatX)
			x /= retain_p
		return x

	# LOSS FUNCTION #
	def categorical_cross_entropy(self, y, y_hat):
		return T.mean(-T.log(y_hat[T.arange(y.shape[0]), y]), axis=0)

	def get_params(self):
		return self.params

	def floatX(self, arr):
		return np.asarray(arr, dtype=theano.config.floatX)

	def init_weights(self, shape, sigma=0.01, name=''):
		if sigma == 0:
			W_bound = np.sqrt(6. / (shape[0] + shape[1]))
			return theano.shared(self.floatX(self.numpy_rng.uniform(low=-W_bound, high=W_bound, size=shape)), borrow=True, name=name)
		return theano.shared(self.floatX(self.numpy_rng.randn(*shape) * sigma), borrow=True, name=name)

	def init_filter(self, filter_shape, input_shape=None, output_shape=None, sigma=0.01, name=''):
		if input_shape is None and output_shape is None:
			# use the heuristic: sample filter weigths from uniform(sqrt(6.0 / (fan_in + fan_out)))
			fan_in = np.prod(input_shape[1:])
			fan_out = np.prod(output_shape[1:])
			W_bound = np.sqrt(6. / (fan_in + fan_out))
			return theano.shared(self.floatX(self.numpy_rng.uniform(low=-W_bound, high=W_bound, size=filter_shape)), borrow=True, name=name)
		else:
			# sample weights from normal distribution with 0 mean and sigma std
			return theano.shared(self.floatX(self.numpy_rng.randn(*filter_shape) * sigma), borrow=True, name=name)

	def sgd(self, cost, params, learning_rate=0.1):
		# compute the gradients of each parameter w.r.t. the loss
		pgrads = T.grad(cost, wrt=params)
		# define the sgd updates
		updates = OrderedDict([(p, p - learning_rate * g) for p, g in zip(params, pgrads)])
		return updates

	def adagrad(self, cost, params, learning_rate=0.1, epsilon=1e-6):
		# compute the gradients of each parameter w.r.t. the loss
		pgrads = T.grad(cost, wrt=params)
		updates = OrderedDict()
		for p, g in zip(params, pgrads):
			acc = theano.shared(p.get_value(borrow=True) * 0., borrow=True)
			acc_new = acc + g ** 2
			g_scaling = T.sqrt(acc_new + epsilon)
			g = g / g_scaling
			updates[acc] = acc_new
			updates[p] = p - learning_rate * g
		return updates

	def rmsprop(self, cost, params, learning_rate=1.0, decay=0.99, epsilon=1e-6):
		# compute the gradients of each parameter w.r.t. the loss
		pgrads = T.grad(cost, wrt=params)
		updates = OrderedDict()
		for p, g in zip(params, pgrads):
			acc = theano.shared(p.get_value(borrow=True) * 0., borrow=True)
			acc_new = decay * acc + (1.0 - decay) * g ** 2
			g_scaling = T.sqrt(acc_new + epsilon)
			g = g / g_scaling
			updates[acc] = acc_new
			updates[p] = p - learning_rate * g
		return updates

	def apply_momentum(self, updates, momentum=0.5):
		updates = OrderedDict(updates)
		for p in updates.keys():
			velocity = theano.shared(p.get_value(borrow=True) * 0., borrow=True)
			# updates[p] = p - learning_rate * dp
			p_new = momentum * velocity + updates[p] 	# p + momentum * velocity - learning_rate * dp
			updates[velocity] = p_new - p 	# momentum * velocity - learning_rate * dp
			updates[p] = p_new 	
		return updates

	def fit(self, x, y, epochs=10, shuffle_training=True):
		if self.train_fn is None:
			print 'Compiling the training functions'
			# the input variable now is a 4d tensor
			X = T.tensor4()
			y_sym = T.ivector()
			# initialize filter and hidden units weights
			self.params = self.init()
			# build the model and the output variables
			y_hat = self.model(X, self.dropout_p_input, self.dropout_p_conv, self.dropout_p_hidden)
			cost = self.categorical_cross_entropy(y_sym, y_hat)
			updates = self.optimization_fn(cost, self.params, self.learning_rate)
			if self.momentum > 0.:
				updates = self.apply_momentum(updates, self.momentum)

			self.train_fn = theano.function(inputs=[X, y_sym], outputs=cost, updates=updates)

		if shuffle_training:
			shuffle_idx = self.numpy_rng.permutation(x.shape[0])
			x = x[shuffle_idx]
			y = y[shuffle_idx]
		num_train_batches = -(-x.shape[0] // self.batch_size)
		cost_history = []
		print 'Training started'
		for e in range(epochs):
			avg_cost = 0
			for bidx in range(num_train_batches):
				batch_x = x[bidx * self.batch_size: (bidx + 1) * self.batch_size]
				batch_y = y[bidx * self.batch_size: (bidx + 1) * self.batch_size]
				batch_cost = self.train_fn(batch_x, batch_y)
				cost_history.append(batch_cost)
				if np.isnan(batch_cost):
					print 'NaN cost detected. Abort'
					return
				avg_cost += batch_cost
			avg_cost /= num_train_batches
			print 'Epoch: {} Loss: {:.8f}'.format(e + 1, avg_cost)
		return cost_history

	def predict(self, x):
		if self.pred_fn is None:
			X = T.tensor4()
			y_hat_pred = self.model(X, 0.0, 0.0, 0.0)
			y_pred = T.argmax(y_hat_pred, axis=1)
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

	# reshape the input data to be compliant with the input shape expected by the CNN
	n_train_x = train_x.shape[0]
	n_valid_x = valid_x.shape[0]
	n_test_x = test_x.shape[0]
	train_x = train_x.reshape(n_train_x, 1, 28, 28)
	valid_x = valid_x.reshape(n_valid_x, 1, 28, 28)
	test_x = test_x.reshape(n_test_x, 1, 28, 28)

	model = ConvNet(
		input_shape=train_x.shape[1:], 
		n_classes=10,
		optim='rmsprop',
		activation='relu',
		num_kernels=[20, 50],
		hidden_layers=[256],
		dropout_p_input=0.0, 
		dropout_p_conv=0.5, 
		dropout_p_hidden=0.5, 
		learning_rate=0.001, 
		momentum=0.0)

	# WARNING: If possible run on CUDA enabled GPU, otherwise it may take a long time to complete on CPU
	t0 = time.time()
	tch = model.fit(train_x, train_y, epochs=25)
	print 'Training completed in {:.2f} sec'.format(time.time() - t0)

	valid_y_pred = model.predict(valid_x)
	valid_accuracy = np.sum(valid_y_pred == valid_y, dtype=np.float32) / valid_y.shape[0]
	print 'Validation accuracy: {:.2f}'.format(valid_accuracy * 100)	# you should get around 99%

	test_y_pred = model.predict(test_x)
	test_accuracy = np.sum(test_y_pred == test_y, dtype=np.float32) / test_y.shape[0]
	print 'Test accuracy: {:.2f}'.format(test_accuracy * 100)	# you should get around 99%

	# dest_dir = 'convnet_train'
	# if not os.path.exists(dest_dir):
	# 	os.makedirs(dest_dir)
	# fname = os.path.join(dest_dir, 
	# 	'convnet_{}_{}_k{}_l{}_lr{}_m{}_di{}_dc{}_dh{}.npz'.format(
	# 	model.optimization,
	# 	model.activation,
	# 	'-'.join(map(str, model.num_kernels)), 
	# 	'-'.join(map(str, model.hidden_layers)), 
	# 	model.learning_rate,
	# 	model.momentum,
	# 	model.dropout_p_input,
	# 	model.dropout_p_conv,
	# 	model.dropout_p_hidden))

	# np.savez_compressed(fname, train_cost=tch, valid_accuracy=valid_accuracy, test_accuracy=test_accuracy)

