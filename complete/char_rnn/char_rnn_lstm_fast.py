# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-03-10 10:54:02
# @Last Modified by:   massimo
# @Last Modified time: 2016-03-10 23:26:40
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils import init_weights, init_ortho, sigmoid, tanh
from updates import adagrad, apply_momentum
import time
from collections import OrderedDict
import os
import cPickle as pkl

class LSTMCharRNN:
	def __init__(
		self,
		rnn_layers=None, 
		batch_size=32,
		grad_clip=5.0,
		learning_rate=0.01,
		momentum=0.5,
		dropout_p_hidden=0.0,
		numpy_rng=None,
		theano_rng=None
		):
		if rnn_layers is None:
			rnn_layers = [100, 100]
		self.rnn_layers = rnn_layers
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.grad_clip = grad_clip
		self.dropout_p_hidden = dropout_p_hidden

		if numpy_rng is None:
			numpy_rng = np.random.RandomState(2**30)
		self.numpy_rng = numpy_rng
		if theano_rng is None:
			theano_rng = RandomStreams(2**30)
		self.theano_rng = theano_rng

		self.vocab, self.params, self.train_fn, self.sample_fn = None, None, None, None


	def init(self):
		# initialze the weigths for the LSTM RNN
		# group Wi, Wf, Wo and Wc into a W matrix with 4 times the original number of columns
		# group bi, bf, bo and bc into a b matrix with 4 times the original number of columns
		# group Whi, Whf, Who and Whc into a Wh matrix with 4 times the original number of columns
		# this allows a better usage of the vector units in the CPU/GPU
		self.W, self.Wh, self.b = [], [], []
		for i in range(len(self.rnn_layers)):
			n_inputs = self.vocab_size if i == 0 else self.rnn_layers[i-1]
			self.W.append(init_weights((n_inputs, 4 * self.rnn_layers[i]), rng=self.numpy_rng, name='W_{}'.format(i)))
			self.Wh.append(init_ortho(self.rnn_layers[i], num_matrices=4, rng=self.numpy_rng, name='Wh_{}'.format(i)))
			self.b.append(theano.shared(value=np.zeros((4 * self.rnn_layers[i],), dtype=theano.config.floatX), borrow=True, name='b_{}'.format(i)))

		self.Why = init_weights((self.rnn_layers[i], self.vocab_size), rng=self.numpy_rng, name='Why')					# hidden to output weights
		self.by = theano.shared(value=np.zeros((self.vocab_size,), dtype=theano.config.floatX), borrow=True, name='by')		# hidden to output bias
		params = self.W + self.Wh + self.b + [self.Why, self.by] 
		return params

	def model(self, X, dropout_p_hidden):
		def _slice(x, n, dim):
			if x.ndim == 3:
				return x[:, :, n * dim: (n + 1) * dim]
			return x[:, n * dim: (n + 1) * dim]

		def _step_index(x_t, ct_1, ht_1, W, Wh, b, dim):
			# x_t: array of type int32
			# use indexing on W matrix instead of computing dot product with the one-hot representation of the input for computational and memory efficiency
			tmp = W[x_t] + T.dot(ht_1, Wh) + b
			i = sigmoid(_slice(tmp, 0, dim))
			f = sigmoid(_slice(tmp, 1, dim))
			o = sigmoid(_slice(tmp, 2, dim))
			c = tanh(_slice(tmp, 3, dim))
			c_new = i * c + f * ct_1
			h_new = o * tanh(c_new)
			return c_new, h_new

		def _step(x_t, ct_1, ht_1, W, Wh, b, dim):
			tmp = T.dot(x_t, W) + T.dot(ht_1, Wh) + b
			i = sigmoid(_slice(tmp, 0, dim))
			f = sigmoid(_slice(tmp, 1, dim))
			o = sigmoid(_slice(tmp, 2, dim))
			c = tanh(_slice(tmp, 3, dim))
			c_new = i * c + f * ct_1
			h_new = o * tanh(c_new)
			return c_new, h_new

		x = X[:-1]
		y = X[1:]
		n_steps, batch_size = x.shape[0], x.shape[1]
		for i in range(len(self.rnn_layers)):
			rval, _ = theano.scan(
				_step_index if i == 0 else _step,
				sequences=[x if i == 0 else h],
				outputs_info=[T.alloc(self.floatX(0.), batch_size, self.rnn_layers[i]), T.alloc(self.floatX(0.), batch_size, self.rnn_layers[i])],		# init c and h matrices to zeros
				non_sequences=[self.W[i], self.Wh[i], self.b[i], np.int32(self.rnn_layers[i])],		# np.int32 forces Theano to use the right data type, otherwise problems with the indexing may occur in runtime
				n_steps=n_steps,
				strict=True)
			h = rval[1]
			h = self.dropout(h, dropout_p_hidden)

		# x has shape (n_steps, batch_size, rnn_layers[-1])
		y_logit = T.dot(h, self.Why) + self.by
		# y_logit has shape (n_steps, batch_size, vocab_size)
		# to compute the softmax properly, we need a 2 dimensional array with vocab_size has second dimension
		y_logit = y_logit.reshape([y_logit.shape[0] * y_logit.shape[1], y_logit.shape[2]])
		y_hat = self.softmax(y_logit)
		loss = self.cross_entropy(y, y_hat)
		return y_hat, loss

	def model_sample(self, X, C, H, sampling_temp):
		def _slice(x, n, dim):
			if x.ndim == 3:
				return x[:, :, n * dim: (n + 1) * dim]
			return x[:, n * dim: (n + 1) * dim]

		def _step_index(x_t, ct_1, ht_1, W, Wh, b, dim):
			# x_t: array of type int32
			# use indexing on W matrix instead of computing dot product with the one-hot representation of the input for computational and memory efficiency
			tmp = W[x_t] + T.dot(ht_1, Wh) + b
			i = sigmoid(_slice(tmp, 0, dim))
			f = sigmoid(_slice(tmp, 1, dim))
			o = sigmoid(_slice(tmp, 2, dim))
			c = tanh(_slice(tmp, 3, dim))
			c_new = i * c + f * ct_1
			h_new = o * tanh(c_new)
			return c_new, h_new

		def _step(x_t, ct_1, ht_1, W, Wh, b, dim):
			tmp = T.dot(x_t, W) + T.dot(ht_1, Wh) + b
			i = sigmoid(_slice(tmp, 0, dim))
			f = sigmoid(_slice(tmp, 1, dim))
			o = sigmoid(_slice(tmp, 2, dim))
			c = tanh(_slice(tmp, 3, dim))
			c_new = i * c + f * ct_1
			h_new = o * tanh(c_new)
			return c_new, h_new

		C_new, H_new = [], []
		for i in range(len(self.rnn_layers)):
			if i == 0:
				c_new, h_new = _step_index(X, C[i], H[i], self.W[i], self.Wh[i], self.b[i], np.int32(self.rnn_layers[i]))
			else:
				c_new, h_new = _step(H_new[i-1], C[i], H[i], self.W[i], self.Wh[i], self.b[i], np.int32(self.rnn_layers[i]))
			C_new.append(c_new)
			H_new.append(h_new)

		h = H_new[-1]
		y_logit = T.dot(h, self.Why) + self.by
		y_logit = y_logit.reshape([y_logit.shape[0] * y_logit.shape[1], y_logit.shape[2]])
		y_hat = self.softmax(y_logit, sampling_temp)
		return y_hat, C_new, H_new

	def cross_entropy(self, y, y_hat):
		# y_hat has shape (n_steps * batch_size, vocab_size)
		# y has shape (n_steps, batch_size)
		y_flatten = y.flatten()
		y_hat_flatten = y_hat.flatten()
		offsets = T.cast(T.arange(y_flatten.shape[0]) * self.vocab_size + y_flatten, 'int32')
		cost = -T.log(y_hat_flatten[offsets])
		cost = cost.reshape([y.shape[0], y.shape[1]])
		cost = T.mean(cost, axis=0)			# compute the average cost per input sequence 
		return T.mean(cost)					# then the average batch cost

	def softmax(self, x, temp=1.0):
		# divide x by the softmax temperature
		x /= temp
		# numerically stable version of softmax (it avoids exp to blowup due to high values of x)
		exp_x = T.exp(x - T.max(x, axis=1, keepdims=True))
		return exp_x / T.sum(exp_x, axis=1, keepdims=True)

	# DROPOUT #
	def dropout(self, x, p=0):
		if p > 0:
			retain_p = 1.0 - x
			x *= self.theano_rng.binomial(x.shape, p=retain_p, dtype=theano.config.floatX)
			x /= retain_p
		return x
		
	def get_params(self):
		return self.params

	def export_params(self):
		return [p.get_value() for p in self.params]

	def import_params(self, iparams):
		assert len(self.params) == len(iparams), 'Different number of params ({} != {})'.format(len(self.params), len(iparams))
		for psrc, pdest in zip(iparams, self.params):
			pdest_shape = pdest.get_value(borrow=True).shape
			assert psrc.shape == pdest_shape, 'Source and destination param shapes do not correspond ({} != {})'.format(psrc.shape, pdest_shape)
			pdest.set_value(psrc, borrow=True)

	def floatX(self, arr):
		return np.asarray(arr, dtype=theano.config.floatX)

	def _split_sequences(self, x_ix, seq_length=25, padding_char=' '):
		# pad the entire text with spaces to make the total length multiple of sequence length
		x_padding = np.asarray([self.ch_to_ix[padding_char]] * (seq_length - x_ix.shape[0] % seq_length), dtype=np.int32)
		x_ix = np.concatenate((x_ix, x_padding))
		# split x into blocks of length equal to seq_length
		n_seqs = int(x_ix.shape[0] / seq_length)
		x_ix = x_ix.reshape((n_seqs, seq_length))
		return x_ix

	def fit(self, x, valid=None, epochs=10, seq_length=25, sampling_temp=0.7, sample_freq=10, checkpoint_freq=10, checkpoints_dir='models', unk_char='*'):
		# NOTE: checkpoints are generated only when a validation set is provided
		# build the character vocabulary
		vocab = set(x)
		if self.vocab is None or vocab != self.vocab:
			self.vocab = vocab
			self.vocab.add(unk_char)			# special placeholder for out-of-vocabulary characters
			self.vocab_size = len(vocab)
			self.ch_to_ix = {ch: i for i, ch in enumerate(vocab)}
			self.ix_to_ch = {v: k for k, v in self.ch_to_ix.iteritems()}
			print 'Vocab size:', self.vocab_size

		# NOTE: checkpoints will be generated only if a validation set is provided
		if self.train_fn is None:
			print 'Compiling the training functions'
			X = T.imatrix()
			self.params = self.init()
			y_hat, cost = self.model(X, self.dropout_p_hidden)
			pgrads = T.grad(cost, wrt=self.params)
			# gradient clipping to avoid exploding gradients
			if self.grad_clip > 0.:
				gnorm = T.sum([T.sum(g ** 2) for g in pgrads])
				# to clip gradients we use the following heuristic
				# new_g = g * grad_clip / total_grad_norm
				pgrads = [T.switch(gnorm > self.grad_clip, g * self.grad_clip / gnorm, g) for g in pgrads]

			updates = adagrad(cost, self.params, grads=pgrads, learning_rate=self.learning_rate)
			if self.momentum > 0.:
				updates = apply_momentum(updates, self.momentum)
			self.train_fn = theano.function(inputs=[X], outputs=cost, updates=updates)
			self.cost_fn = theano.function(inputs=[X], outputs=cost)

		# convert strings to integer vectors
		x_ix = np.asarray([self.ch_to_ix[ch] for ch in x], dtype=np.int32)
		if valid is not None:
			valid_ix = np.asarray([self.ch_to_ix.get(ch, self.ch_to_ix[unk_char]) for ch in valid], dtype=np.int32)
			if not os.path.exists(checkpoints_dir):
				os.makedirs(checkpoints_dir)

		# Let's check the initial cost matches the exected one
		# print 'Expected initial cost:', np.log(len(vocab))
		# print 'Actual initial cost:', self.cost_fn(x_ix[:,None])
		 
		# split the training sequence into equal blocks of length seq_length
		x_ix = self._split_sequences(x_ix, seq_length, padding_char=' ')
		# randomly the training sequences
		x_ix = x_ix[self.numpy_rng.permutation(x_ix.shape[0])]

		# then start training
		num_train_batches = -(-x_ix.shape[0] // self.batch_size)
		print 'Training started'
		train_cost_history = []
		if valid is not None:
			valid_cost_history = []
		for e in range(epochs):
			avg_cost = 0
			for bidx in range(num_train_batches):
				batch_x = x_ix[bidx * self.batch_size: (bidx + 1) * self.batch_size]
				batch_cost = self.train_fn(batch_x.transpose([1, 0]))
				train_cost_history.append(float(batch_cost))
				if np.isnan(batch_cost):
					print 'NaN cost detected. Abort'
					return
				avg_cost += batch_cost
			avg_cost /= num_train_batches
			if valid is not None:
				valid_cost = float(self.cost_fn(valid_ix[:, None]))
				valid_cost_history.append(valid_cost)
				print 'Epoch: {} Train Loss: {:.4f} Valid Loss: {:.4f}'.format(e, avg_cost, valid_cost)
				if checkpoint_freq > 0 and (e + 1) % checkpoint_freq == 0:
					# pickle to save the current state of training
					chk_path = os.path.join(checkpoints_dir, 'charrnn_lstm_fast_{}_epoch{}_t{:.4f}_v{:.4f}.pkl'.format(len(self.rnn_layers), e, avg_cost, valid_cost))
					state = {
						'epoch': e,
						'train_cost_history': train_cost_history,
						'valid_cost_history': valid_cost_history,
						'train_cost': 	avg_cost,
						'valid_cost': 	valid_cost,
						'params': 		self.export_params(),
						'vocab':		self.vocab,
						'rnn_layers':	self.rnn_layers,
						'batch_size':	self.batch_size,
						'learning_rate':	self.learning_rate,
						'dropout_p_hidden':	self.dropout_p_hidden,
						'momentum':		self.momentum,
						'grad_clip':	self.grad_clip,
					}
					pkl.dump(state, open(chk_path, 'wb'), pkl.HIGHEST_PROTOCOL)
					print 'Written checkpoint:', chk_path
			else:
				print 'Epoch: {} Train Loss: {:.4f}'.format(e + 1, avg_cost)
			if (e + 1) % sample_freq == 0:
				print '\nSampled string:\n{}\n'.format(self.sample(seed_string=''))

	def init_from(self, checkpoint):
		state = pkl.load(open(checkpoint, 'rb'))
		# import the vocabulary
		self.vocab = state['vocab']
		self.ch_to_ix = {ch: i for i, ch in enumerate(self.vocab)}
		self.ix_to_ch = {v: k for k, v in self.ch_to_ix.iteritems()}
		self.vocab_size = len(self.vocab)
		# import the network configuration
		self.rnn_layers = state['rnn_layers']
		self.batch_size = state['batch_size']
		self.learning_rate = state['learning_rate']
		self.dropout_p_hidden = state['dropout_p_hidden']
		self.momentum = state['momentum']
		self.grad_clip = state['grad_clip']
		# import the network parameters
		self.params = self.init()
		self.import_params(state['params'])
		self.train_fn, self.sample_fn = None, None

	def sample(self, sample_length=100, sampling_temp=1.0, seed_string='', use_sampling=True):
		if self.params is None:
			print 'Run fit() or init_from() before sampling'
			return
		if self.sample_fn is None:
			# symbolic variable for input character and the softmax temperature
			x = T.ivector()
			temp = T.scalar()
			# store the state of the rnn into a dedicated shared variable per layer
			self.H, self.C = [], []
			for i in range(len(self.rnn_layers)):
				self.C.append(theano.shared(value=np.zeros((1, 1, self.rnn_layers[i]), dtype=theano.config.floatX), borrow=True, name='C_{}'.format(i)))
				self.H.append(theano.shared(value=np.zeros((1, 1, self.rnn_layers[i]), dtype=theano.config.floatX), borrow=True, name='H_{}'.format(i)))
			# build the sampler
			y_hat_sample, C_new, H_new = self.model_sample(x, self.C, self.H, temp)
			# update each hidden state in the rnn
			updates = OrderedDict()
			for i in range(len(self.rnn_layers)):
				updates[self.C[i]] = C_new[i]
				updates[self.H[i]] = H_new[i]
			# define the sampling function
			self.sample_fn = theano.function(inputs=[x, temp], outputs=y_hat_sample, updates=updates)

		# ensure that the rnn state is set to zero
		for i in range(len(self.rnn_layers)):
			self.C[i].set_value(np.zeros((1, 1, self.rnn_layers[i]), dtype=theano.config.floatX), borrow=True)
			self.H[i].set_value(np.zeros((1, 1, self.rnn_layers[i]), dtype=theano.config.floatX), borrow=True)
		if len(seed_string) == 0:
			# start from an uniformly sampled character
			seed_string = self.ix_to_ch[self.numpy_rng.randint(self.vocab_size)]
		# bootstrap the rnn using the seed_string
		for ch in seed_string:
			next_char_proba = self.sample_fn([self.ch_to_ix[ch]], sampling_temp).squeeze()
		sampled_str = []
		# start the real sampling from the rnn
		for i in range(sample_length):
			if use_sampling:
				ch_ix = self.numpy_rng.choice(self.vocab_size, p=next_char_proba)
			else:
				ch_ix = np.argmax(next_char_proba)
			sampled_str.append(self.ix_to_ch[ch_ix])
			next_char_proba = self.sample_fn([ch_ix], sampling_temp).squeeze()
		return seed_string + ''.join(sampled_str)

from sys import argv
if __name__ == '__main__':
	input_file = argv[1]
	x = open(input_file, 'r').read()

	train_size = int(0.9 * len(x))
	valid_size = int(0.05 * len(x))
	train_x = x[:train_size]
	valid_x = x[train_size:train_size+valid_size]
	test_x = x[train_size+valid_size:]

	checkpoints_dir = os.path.splitext(os.path.basename(input_file))[0]

	model = LSTMCharRNN(
		rnn_layers=[256, 256], 
		batch_size=100,
		grad_clip=5.0,
		dropout_p_hidden=0.0,
		learning_rate=0.01,
		momentum=0.0)

	t0 = time.time()
	model.fit(train_x, valid=valid_x, epochs=100, seq_length=100, checkpoints_dir=checkpoints_dir)
	print 'Training completed in {:.2f} sec'.format(time.time() - t0)




