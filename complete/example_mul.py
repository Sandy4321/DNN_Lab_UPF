# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-03-10 16:55:53
# @Last Modified by:   massimo
# @Last Modified time: 2016-03-10 23:26:44
# Theano imports
import theano
import theano.tensor as T

# Theano symbolic variables (float scalars)
x = T.scalar()
y = T.scalar()

# Our model
z = x * y

# COMPILING into a python function 
mul_fn = theano.function(inputs=[x, y], outputs=z)

# use the function
print mul_fn(2, 5)	# 10.0
print mul_fn(10, 5)	# 50.0