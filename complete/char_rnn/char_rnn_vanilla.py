# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-03-10 23:19:24
# @Last Modified by:   massimo
# @Last Modified time: 2016-03-10 23:26:38
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import utils
from updates import adagrad, apply_momentum
import time
from collections import OrderedDict
import os
import cPickle as pkl

class VanillaCharRNN:
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
		self.dropout_p_hidden = dropout_p_hidden
		self.momentum = momentum
		self.grad_clip = grad_clip

		if numpy_rng is None:
			numpy_rng = np.random.RandomState(2**30)
		self.numpy_rng = numpy_rng
		if theano_rng is None:
			theano_rng = RandomStreams(2**30)
		self.theano_rng = theano_rng

		self.vocab, self.params, self.train_fn, self.sample_fn = None, None, None, None

	def init(self):
		# initialze the weigths for the Vanilla RNN
		self.Wxh, self.Whh, self.bh, self.by = [], [], [], []
		for i in range(len(self.rnn_layers)):
			n_inputs = self.vocab_size if i == 0 else self.rnn_layers[i-1]

			self.Wxh.append(utils.init_weights((n_inputs, self.rnn_layers[i]),
			 rng=self.numpy_rng, name='Wxh_{}'.format(i)))		# input to hidden weigths
			self.Whh.append(utils.init_ortho(self.rnn_layers[i],
			 rng=self.numpy_rng, name='Whh_{}'.format(i)))					# hidden to hidden weights
			self.bh.append(theano.shared(value=np.zeros((self.rnn_layers[i],),
			 dtype=theano.config.floatX), borrow=True, name='bh_{}'.format(i)))		# hidden to hidden bias

		self.Why = utils.init_weights((self.rnn_layers[i], self.vocab_size),
		 rng=self.numpy_rng, name='Why')					# hidden to output weights
		self.by = theano.shared(value=np.zeros((self.vocab_size,),
		 dtype=theano.config.floatX), borrow=True, name='by')		# hidden to output bias
		params = self.Wxh + self.Whh + self.bh + [self.Why, self.by] 
		return params

	def model(self, X, dropout_p_hidden):
		def _step_index(x_t, ht_1, Wxh, Whh, bh):
			# x_t: array of type int32
			# use indexing on Wxh matrix instead of hte dot product of the one-hot vectors for computational and memory efficiency
			return T.tanh(Wxh[x_t] + T.dot(ht_1, Whh) + bh)				# compute the next hidden state

		def _step(x_t, ht_1, Wxh, Whh, bh):
			return T.tanh(T.dot(x_t, Wxh) + T.dot(ht_1, Whh) + bh)		# compute the next hidden state

		x = X[:-1]
		y = X[1:]
		n_steps, batch_size = x.shape[0], x.shape[1]
		for i in range(len(self.rnn_layers)):
			x, _ = theano.scan(
				_step_index if i == 0 else _step,
				sequences=[x],
				outputs_info=T.alloc(self.floatX(0.), batch_size, self.rnn_layers[i]),	# init to zeros every hidden layer in the rnn
				non_sequences=[self.Wxh[i], self.Whh[i], self.bh[i]],
				n_steps=n_steps,
				strict=True)
			x = self.dropout(x, dropout_p_hidden)
		h = x
		# x has shape (n_steps, batch_size, rnn_layers[-1])
		y_logit = T.dot(h, self.Why) + self.by
		# y_logit has shape (n_steps, batch_size, vocab_size)
		# to compute the softmax properly, we need a 2 dimensional array with vocab_size has second dimension
		y_logit = y_logit.reshape([y_logit.shape[0] * y_logit.shape[1], y_logit.shape[2]])
		y_hat = self.softmax(y_logit)
		cost = self.cross_entropy(y, y_hat)
		return y_hat, cost

	def model_sample(self, X, H, sampling_temp):
		def _step_index(x_t, ht_1, Wxh, Whh, bh):
			# x_t: array of type int32
			# use indexing on Wxh matrix instead of hte dot product of the one-hot vectors for computational and memory efficiency
			return T.tanh(Wxh[x_t] + T.dot(ht_1, Whh) + bh)				# compute the next hidden state

		def _step(x_t, ht_1, Wxh, Whh, bh):
			return T.tanh(T.dot(x_t, Wxh) + T.dot(ht_1, Whh) + bh)		# compute the next hidden state

		H_new = []
		for i in range(len(self.rnn_layers)):
			if i == 0:
				H_new.append(_step_index(X, H[i], self.Wxh[i], self.Whh[i], self.bh[i]))
			else:
				H_new.append(_step(H_new[i-1], H[i], self.Wxh[i], self.Whh[i], self.bh[i]))

		h = H_new[-1]
		y_logit = T.dot(h, self.Why) + self.by
		y_logit = y_logit.reshape([y_logit.shape[0] * y_logit.shape[1], y_logit.shape[2]])
		y_hat = self.softmax(y_logit, sampling_temp)
		return y_hat, H_new

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
			self.vocab.add(unk_char)		# special placeholder for out-of-vocabulary characters
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
					chk_path = os.path.join(checkpoints_dir, 'charrnn_vanilla_{}_epoch{}_t{:.4f}_v{:.4f}.pkl'.format(len(self.rnn_layers), e, avg_cost, valid_cost))
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
			# symbolic variable for the softmax temperature
			x = T.ivector()
			temp = T.scalar()
			# store the state of the rnn into a dedicated shared variable per layer
			self.H = []
			for i in range(len(self.rnn_layers)):
				self.H.append(theano.shared(value=np.zeros((1, 1, self.rnn_layers[i]), dtype=theano.config.floatX), borrow=True, name='H_{}'.format(i)))
			# build the sampler
			y_hat_sample, H_new = self.model_sample(x, self.H, temp)
			# update each hidden state in the rnn
			updates = OrderedDict()
			for i in range(len(self.rnn_layers)):
				updates[self.H[i]] = H_new[i]
			# define the sampling function
			self.sample_fn = theano.function(inputs=[x, temp], outputs=y_hat_sample, updates=updates)

		# ensure that the rnn state is set to zero
		for i in range(len(self.rnn_layers)):
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

	model = VanillaCharRNN(
		rnn_layers=[256, 256], 
		batch_size=100,
		grad_clip=5.0,
		dropout_p_hidden=0.0,
		learning_rate=0.01,
		momentum=0.5)

	t0 = time.time()
	model.fit(train_x, valid=valid_x, epochs=100, seq_length=100, checkpoints_dir=checkpoints_dir)
	print 'Training completed in {:.2f} sec'.format(time.time() - t0)



