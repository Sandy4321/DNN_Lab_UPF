# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-03-10 15:33:49
# @Last Modified by:   massimo
# @Last Modified time: 2016-03-10 23:29:56
import theano
import theano.tensor as T
from collections import OrderedDict

def grad_clip(grads, grad_cap=0.5):
	gnorm = T.sum([g ** 2 for g in grads])
	grads_new = [T.switch(gnorm > grad_cap, g * grad_cap / gnorm, g) for g in grads]
	return grads_new

def sgd(cost, params, grads=None, learning_rate=0.1):
	if grads is None:
		# compute the gradients of each parameter w.r.t. the loss
		grads = T.grad(cost, wrt=params)
	# define the sgd updates
	updates = OrderedDict([(p, p - learning_rate * g) for p, g in zip(params, grads)])
	return updates

def adagrad(cost, params, grads=None, learning_rate=0.1, epsilon=1e-6):
	if grads is None:
		# compute the gradients of each parameter w.r.t. the loss
		grads = T.grad(cost, wrt=params)
	updates = OrderedDict()
	for p, g in zip(params, grads):
		acc = theano.shared(p.get_value(borrow=True) * 0., borrow=True)
		acc_new = acc + g ** 2
		g_scaling = T.sqrt(acc_new + epsilon)
		g = g / g_scaling
		updates[acc] = acc_new
		updates[p] = p - learning_rate * g
	return updates

def rmsprop(cost, params, grads=None, learning_rate=0.1, decay=0.99, epsilon=1e-6):
	if grads is None:
		# compute the gradients of each parameter w.r.t. the loss
		grads = T.grad(cost, wrt=params)
	updates = OrderedDict()
	for p, g in zip(params, grads):
		acc = theano.shared(p.get_value(borrow=True) * 0., borrow=True)
		acc_new = decay * acc + (1.0 - decay) * g ** 2
		g_scaling = T.sqrt(acc_new + epsilon)
		g = g / g_scaling
		updates[acc] = acc_new
		updates[p] = p - learning_rate * g
	return updates

def apply_momentum(updates, momentum=0.5):
	updates = OrderedDict(updates)
	for p in updates.keys():
		velocity = theano.shared(p.get_value(borrow=True) * 0., borrow=True)
		# updates[p] = p - learning_rate * dp
		p_new = momentum * velocity + updates[p] 	# p + momentum * velocity - learning_rate * dp
		updates[velocity] = p_new - p 	# momentum * velocity - learning_rate * dp
		updates[p] = p_new 	
	return updates
