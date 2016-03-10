# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-03-10 15:33:49
# @Last Modified by:   massimo
# @Last Modified time: 2016-03-10 23:29:57
import numpy as np
import os
from collections import OrderedDict
from mlp import MLP
from data_loader import load_mnist
import time
import sys

# Minimal code version for the random hyper-parameter optimization
# Makes a fixed number of trials by picking a random combination of the hyper-parameters of the model
# This version was written explicitly for the MLP class described in mpl.py
# but it can be easily adapted to any kind of model having a set_params method
dest_dir = 'opt_logs'
if not os.path.exists(dest_dir):
	os.makedirs(dest_dir)

mnist_path = sys.argv[1]
dataset = load_mnist(mnist_path)
train_x, train_y = dataset[0]
valid_x, valid_y = dataset[1]
test_x, test_y = dataset[2]

params = {
	'learning_rate': [0.1, 0.05, 0.01],
	'momentum': [0.0, 0.5, 0.9, 0.95],
	'dropout_p_input': [0.0, 0.5],
	'dropout_p_hidden': [0.0, 0.5],
}

if not os.path.exists(dest_dir):
	os.makedirs(dest_dir)
num_trials = 10
n_tried = 0
best_valid, best_test, best_conf = None, None, None

while n_tried < num_trials:
	# choose randomly a configuration for the paramters
	try_this = [np.random.randint(len(p)) for p in params.values()]
	try_params = OrderedDict([(k, v[try_this[i]]) for i, (k, v) in enumerate(params.items())])

	model = MLP(
		n_classes=10,
		optim='adagrad',
		n_inputs=train_x.shape[1],
		activation='relu',
		layers=[256, 128])
	model.set_params(**try_params)

	fname = os.path.join(dest_dir, 
		'mlp_{}_{}_l{}_lr{}_m{}_di{}_dh{}.npz'.format(
		model.optimization,
		model.activation,
		'-'.join(map(str, model.layers)), 
		model.learning_rate,
		model.momentum,
		model.dropout_p_input,
		model.dropout_p_hidden))

	# check if this configuration has already been tried
	if os.path.exists(fname):
		continue
	# if not continue with training
	print 'Trying the following configuration:', try_params
	t0 = time.time()
	train_cost_history = model.fit(train_x, train_y, epochs=25)
	print 'Training completed in {:.2f} sec'.format(time.time() - t0)

	# and validation
	valid_y_pred = model.predict(valid_x)
	valid_accuracy = np.sum(valid_y_pred == valid_y, dtype=np.float32) / valid_y.shape[0]
	print 'Validation accuracy: {:.2f}'.format(valid_accuracy * 100)

	if best_valid is None or best_valid < valid_accuracy:
		best_valid = valid_accuracy
		best_conf = try_params
		test_y_pred = model.predict(test_x)
		best_test = np.sum(test_y_pred == test_y, dtype=np.float32) / test_y.shape[0]

	# finally save the training costs and the validation accuracy to disk
	np.savez_compressed(fname, train_cost=train_cost_history, valid_accuracy=valid_accuracy)
	n_tried += 1

print 'Best configuration:', best_conf
print 'Best validation accuracy: {:.2f}'.format(best_valid * 100)
print 'Best test accuracy: {:.2f}'.format(best_test * 100)