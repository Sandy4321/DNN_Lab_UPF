# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-03-10 23:22:09
# @Last Modified by:   massimo
# @Last Modified time: 2016-03-10 23:28:00
from sys import argv
import numpy as np

# usage: python sampler.py <char-rnn-model> <checkpoint> <seed-string> <softmax-temp> <rnd-seed>
# example: python sampler.py vanilla obama_speechs/charrnn_vanilla_2_epoch19_t1.3232_v1.3087.pkl 'Hello America' 0.6 1000 5

model = argv[1]
if model == 'vanilla':
	from char_rnn_vanilla import VanillaCharRNN as CharRNN
elif model == 'lstm':
	from char_rnn_lstm import LSTMCharRNN as CharRNN
elif model == 'lstm_fast':
	from char_rnn_lstm_fast import LSTMCharRNN as CharRNN
else:
	raise RuntimeError('Unknown model "{}"'.format(model))

checkpoint = argv[2]
seed_string = argv[3]
temperature = float(argv[4])
sample_length = int(argv[5])
rnd_seed = int(argv[6])
numpy_rng = np.random.RandomState(rnd_seed)

model = CharRNN(numpy_rng=numpy_rng)
model.init_from(checkpoint)
sampled_string = model.sample(sample_length=sample_length, sampling_temp=temperature, seed_string=seed_string, use_sampling=True)
print sampled_string