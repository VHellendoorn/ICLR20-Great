import numpy as np
import tensorflow as tf

# Based on https://github.com/DongjunLee/transformer-tensorflow/blob/master/transformer/attention.py
def positional_encoding(dim, sentence_length, dtype=tf.float32):
	encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
	encoded_vec[::2] = np.sin(encoded_vec[::2])
	encoded_vec[1::2] = np.cos(encoded_vec[1::2])
	return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

def prefix_sum(arr):
	res = [0]
	for a in arr: res.append(res[-1] + a)
	return res
