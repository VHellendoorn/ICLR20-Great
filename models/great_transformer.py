import tensorflow as tf

from . import util


class AttentionLayer(tf.keras.layers.Layer):
	"""
		Implementation of multi-headed attention with optional edge-bias.
		
		This class supports self-attention and key-value attention, with (non-optional) masks. If bias_dim is not None, the attention computation(s) assumes that a (sparse) bias vector is provided, formatted like: (edge_type, batch_index, key_index, query_index). Bias edge types are embedded in the same dimension as each head's attention and projected to a scalar before being inserted into the attention computation as (q + b) * k.
	"""
	
	def __init__(self, attention_dim, num_heads=None, hidden_dim=None, bias_dim=None):
		super(AttentionLayer, self).__init__()
		self.attention_dim = attention_dim
		self.hidden_dim = hidden_dim if hidden_dim is not None else self.attention_dim
		self.num_heads = 1 if num_heads is None else num_heads
		self.attention_dim_per_head = self.attention_dim // self.num_heads
		self.bias_dim = bias_dim
	
	def build(self, _):
		self.attn_query = self.add_weight(name='q', shape=(self.hidden_dim, self.num_heads, self.attention_dim_per_head), initializer="glorot_uniform")
		self.attn_keys = self.add_weight(name='k', shape=(self.hidden_dim, self.num_heads, self.attention_dim_per_head), initializer="glorot_uniform")
		self.attn_values = self.add_weight(name='v', shape=(self.hidden_dim, self.num_heads, self.attention_dim_per_head), initializer="glorot_uniform")
		self.weight_out = self.add_weight(name='o', shape=(self.num_heads, self.attention_dim_per_head, self.hidden_dim), initializer="glorot_uniform")
		if self.bias_dim is not None:
			self.bias_embs = self.add_weight(name='e1', shape=(self.bias_dim, self.attention_dim_per_head), initializer="glorot_uniform")
			self.bias_scalar = self.add_weight(name='e2', shape=(self.attention_dim_per_head, 1), initializer="glorot_uniform")
	
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, 4), dtype=tf.int32)])
	def call(self, states, key_states, masks, attention_bias):
		# Compute key, query and value vectors, reshaped to [Batch, Heads, Time, Dim] where Dim is attention_dim//num_heads.
		query, keys, values = self.compute_qkv(states, key_states)
		
		# Compute attention weights, and context from these.
		alpha = self.get_attention_weights(query, keys, masks, attention_bias)
		
		# Compute weigthed context and project out.
		context = tf.einsum('bhqk,bkha->bqha', alpha, values)
		context = tf.einsum('btha,had->btd', context, self.weight_out)
		return context
	
	# Compute key, query and value vectors.
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)])
	def compute_qkv(self, states, key_states):
		query = tf.einsum('btd,dha->btha', states, self.attn_query) # Queries are always computed on states
		keys = tf.einsum('btd,dha->btha', key_states, self.attn_keys)
		values = tf.einsum('btd,dha->btha', key_states, self.attn_values)
		return query, keys, values
	
	# Compute attention weights from cross-product between keys and queries (scaled, masked, softmaxed).
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, 4), dtype=tf.int32)])
	def get_attention_weights(self, query, keys, masks, attention_bias):
		alpha = tf.einsum('bkha,bqha->bhqk', keys, query)
		
		# If bias_dim is set, assume that a bias vector is provided.
		if self.bias_dim is not None:
			# Embed edge types in per-head-attention dimension. Experimentally, mat-mul tends to be faster here, but regular embedding is equally valid.
			bias = tf.matmul(tf.one_hot(attention_bias[:, 0], self.bias_dim), self.bias_embs)
			# Project down to a scalar.
			bias = tf.squeeze(tf.matmul(bias, self.bias_scalar), -1)
			alpha_shape = tf.shape(alpha)
			bias_shape = tf.stack([alpha_shape[0], alpha_shape[2], alpha_shape[3]])
			# Scatter edge biases to their [batch_index, key_index, query_index] positions.
			bias = tf.scatter_nd(attention_bias[:, 1:], bias, bias_shape)
			
			# Since bias is a scalar, we can reduce memory cost by rewriting the attention from (q + b) * k to q*k + b*reduce_sum(k, -1)
			summed_keys = tf.reduce_sum(keys, -1) # bkh
			bias = tf.einsum('bqk,bkh->bhqk', bias, summed_keys)
			# Accordingly, simply add the bias as a residual to standard dot-product attention.
			alpha += bias
		
		# Scale and apply mask
		alpha *= tf.math.rsqrt(tf.cast(self.attention_dim_per_head, "float32"))
		alpha = alpha * masks + (1.0 - tf.math.ceil(masks)) * tf.float32.min
		alpha = tf.nn.softmax(alpha)
		alpha *= masks
		return alpha

class LayerNormalization(tf.keras.layers.Layer):
	def __init__(self, hidden_dim):
		super(LayerNormalization, self).__init__()
		self.hidden_dim = hidden_dim
	
	def build(self, _):	
		self.scale = tf.Variable(tf.ones(self.hidden_dim))
		self.bias = tf.Variable(tf.zeros(self.hidden_dim))
		self.build = True
	
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32)])
	def call(self, x, epsilon=1e-3):
		mean, variance = tf.nn.moments(x, -1, keepdims=True)
		norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
		return norm_x * self.scale + self.bias

class Transformer(tf.keras.layers.Layer):
	"""Transformer language model: converts indices into hidden states through layers of multi-headed attention and feed-forward dense layers.
	
		Augments a generic Transformer with attentional bias, if bias_dim is provided. See documentation on AttentionLayer for more details.
		To generate language from the resulting states, pass the states to the "predict" function. Note that it assumes that the input vocabulary is output vocabulary (i.e., it reuses the model's embedding table).
	"""
	NOOP_BIAS = tf.zeros((0, 4), 'int32')
	
	def __init__(self, model_config, bias_dim=None, shared_embedding=None, vocab_dim=None, is_encoder_decoder=False):
		super(Transformer, self).__init__()
		self.bias_dim = bias_dim
		self.is_encoder_decoder = is_encoder_decoder
		self.hidden_dim = model_config["hidden_dim"]
		self.ff_dim = model_config["ff_dim"]
		self.attention_dim = model_config["attention_dim"]
		self.num_layers = model_config["num_layers"]
		self.num_heads = model_config["num_heads"]
		self.dropout_rate = model_config["dropout_rate"]

		# Initialize embedding variable in constructor to allow reuse by other models
		if shared_embedding is not None:
			self.embed = shared_embedding
		elif vocab_dim is None:
			raise ValueError("Pass either a vocabulary dimension or an embedding Variable")
		else:
			random_init = tf.random_normal_initializer(stddev=self.hidden_dim ** -0.5)
			self.embed = tf.Variable(random_init([vocab_dim, self.hidden_dim]), dtype=tf.float32)
		
		# Initialize default positional encoding for very long sequences. Can make this a parameter if necessary.
		self.pos_enc = tf.constant(util.positional_encoding(self.hidden_dim, 5000))
	
	def build(self, _):
		# Set up multi-headed attention, and feed-forward layers.
		make_att = lambda : AttentionLayer(self.attention_dim, self.num_heads, self.hidden_dim, self.bias_dim)
		self.attention = [make_att() for _ in range(self.num_layers)]#make_att_deprecated
		if self.is_encoder_decoder:
			self.enc_attention = [make_att() for _ in range(self.num_layers)]
		
		# Layer normalization for every residual layer
		self.ln = [[LayerNormalization(self.hidden_dim) for _ in range(3 if self.is_encoder_decoder else 2)] for _ in range(self.num_layers)]
		self.ln_out = LayerNormalization(self.hidden_dim)
		
		# Two-layer feed-forward with wide layer in the middle
		self.ff_1 = [tf.keras.layers.Dense(self.ff_dim, activation="relu") for _ in range(self.num_layers)]
		self.ff_2 = [tf.keras.layers.Dense(self.hidden_dim) for _ in range(self.num_layers)]
	
	# Default 'call' applies standard self-attention, with dropout if training=True.
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, 4), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.bool)])
	def call(self, states, masks, attention_bias, training):
		real_dropout_rate = self.dropout_rate * tf.cast(training, 'float32')  # Easier for distributed training than an explicit conditional
		for ix in range(self.num_layers):
			new_states = self.ln[ix][0](states)
			new_states = self.attention[ix](new_states, new_states, masks, attention_bias)
			new_states = tf.nn.dropout(new_states, rate=real_dropout_rate)
			states += new_states
			
			new_states = self.ff_1[ix](self.ln[ix][1](states))
			new_states = tf.nn.dropout(new_states, rate=real_dropout_rate)
			new_states = self.ff_2[ix](new_states)
			new_states = tf.nn.dropout(new_states, rate=real_dropout_rate)
			states += new_states
		return self.ln_out(states)
	
	# Standard encoder-decoder attention, with dropout if training=True.
	# NOTE: tentatively does not support attention bias within the query itself; extending this should be straightforward.
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, 4), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.bool)])
	def enc_dec_attention(self, states, masks, key_states, key_masks, attention_bias, training):
		real_dropout_rate = self.dropout_rate * tf.cast(training, 'float32')  # Easier for distributed training than an explicit conditional
		for ix in range(self.num_layers):
			new_states = self.ln[ix][0](states)
			new_states = self.attention[ix](new_states, new_states, masks, tf.zeros((0,4), dtype='int32'))
			new_states = tf.nn.dropout(new_states, rate=real_dropout_rate)
			states += new_states

			new_states = self.ln[ix][2](states)
			new_states = self.enc_attention[ix](new_states, key_states, key_masks, attention_bias)
			new_states = tf.nn.dropout(new_states, rate=real_dropout_rate)
			states += new_states
			
			new_states = self.ff_1[ix](self.ln[ix][1](states))
			new_states = tf.nn.dropout(new_states, rate=real_dropout_rate)
			new_states = self.ff_2[ix](new_states)
			new_states = tf.nn.dropout(new_states, rate=real_dropout_rate)
			states += new_states
		return self.ln_out(states)

	# Embed inputs. Note: applies scaling before positional encoding.
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
	def embed_inputs(self, inputs):
		states = tf.nn.embedding_lookup(self.embed, inputs)
		states *= tf.math.sqrt(tf.cast(tf.shape(states)[-1], "float32"))
		states += self.pos_enc[:tf.shape(states)[1]]
		return states
	
	# Generates tokens from transformer states using the transposed embedding layer.
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)])
	def predict(self, states):
		return tf.matmul(states, self.embed, transpose_b=True)
	
	# Convenience function: returns a sequence mask in which each token can only see states up to its own position. Useful for generative language modeling (e.g. decoding).
	@tf.function
	def get_sequence_mask(self, seq_len):
		return tf.sequence_mask(lengths=tf.range(1, seq_len + 1), maxlen=seq_len, dtype=tf.float32)
