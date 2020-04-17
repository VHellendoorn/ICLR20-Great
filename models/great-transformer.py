import tensorflow as tf
import util


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
	
	def call(self, states, masks, attention_bias):
		# For simplicity, assume that this is self-attention if the states are a tuple of one element, and encoder-decoder attention otherwise.
		if len(states) == 2:
			states, key_states = states
		else:
			states, key_states = states[0], states[0]
		return self.call_internal(states, key_states, masks, attention_bias)
	
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, 4), dtype=tf.int32)])
	def call_internal(self, states, key_states, masks, attention_bias):
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
		keys = tf.einsum('btd,dha->btha', states if key_states is None else key_states, self.attn_keys)
		values = tf.einsum('btd,dha->btha', states if key_states is None else key_states, self.attn_values)
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
			keys = tf.reduce_sum(keys, -1) # bkh
			bias = tf.einsum('bqk,bkh->bhqk', bias, keys)
			# Accordingly, simply add the bias as a residual to standard dot-product attention.
			alpha += bias
		
		# Scale and apply mask
		alpha *= tf.math.rsqrt(tf.cast(self.attention_dim_per_head, "float32"))
		if masks is not None:
			alpha = alpha * masks + (1.0 - tf.math.ceil(masks)) * tf.float32.min
		alpha = tf.nn.softmax(alpha)
		return alpha

class Transformer(tf.keras.layers.Layer):
	"""Transformer language model: converts indices into hidden states through layers of multi-headed attention and feed-forward dense layers.
	
		Augments a generic Transformer with attentional bias, if bias_dim is provided. See documentation on AttentionLayer for more details.
		To generate language from the resulting states, pass the states to the "predict" function. Note that it assumes that the input vocabulary is output vocabulary (i.e., it reuses the model's embedding table).
	"""
	
	def __init__(self, model_config, vocab_dim, shared_embedding=None, bias_dim=None):
		super(Transformer, self).__init__()
		self.vocab_dim = vocab_dim
		self.bias_dim = bias_dim

		self.embed_dim = model_config["embed_dim"]
		self.hidden_dim = model_config["hidden_dim"]
		assert self.embed_dim == self.hidden_dim, "Embedding and hidden dimension must be equal for Transformer."
		self.ff_dim = model_config["ff_dim"]
		self.attention_dim = model_config["attention_dim"]
		self.num_layers = model_config["num_layers"]
		self.num_heads = model_config["num_heads"]
		self.dropout_rate = model_config["dropout_rate"]

		# Allow reuse of an existing embedding variable, if provided. 
		self.embed = shared_embedding
		# Initialize default positional encoding for very long sequences. Can make this a parameter if necessary.
		self.pos_enc = tf.constant(util.positional_encoding(model_config["embed_dim"], 5000))
	
	def build(self, _):
		# Set up embedding, multi-headed attention, and feed-forward layers.
		if self.embed is None:
			random_init = tf.random_normal_initializer(stddev=self.hidden_dim ** -0.5)
			self.embed = tf.Variable(random_init([self.vocab_dim, self.embed_dim]), dtype=tf.float32)
		
		make_att = lambda : AttentionLayer(self.attention_dim, self.num_heads, self.hidden_dim, self.bias_dim)
		self.attention = [make_att() for _ in range(self.num_layers)]#make_att_deprecated
		self.enc_attention = [make_att() for _ in range(self.num_layers)]
		
		# Layer normalization for every residual layer
		self.ln = [[tf.keras.layers.LayerNormalization() for _ in range(3)] for _ in range(self.num_layers)]
		self.ln_out = tf.keras.layers.LayerNormalization()
		
		# Two-layer feed-forward with wide layer in the middle
		self.ff_1 = [tf.keras.layers.Dense(self.ff_dim, activation="relu") for _ in range(self.num_layers)]
		self.ff_2 = [tf.keras.layers.Dense(self.hidden_dim) for _ in range(self.num_layers)]
	
	def call(self, inputs, masks, training, attention_bias=None):
		# For convenience, packs self-attention and encoder-decoder attention into one implementation; if two inputs are provided, we assume that the first element are the target-domain indices and the second are the encoded states (and masks correspondingly).
		is_enc_dec = len(inputs) == 2
		if is_enc_dec:
			inputs, key_states = inputs
			masks, key_masks = masks
		else:
			inputs = inputs[0]
			masks = masks[0]
		
		# If missing, create a dummy attention-bias just to satisfy input signatures; won't affect anything.
		if attention_bias is None:
			attention_bias = tf.zeros((0,4), dtype='int32')
		else:
			# Edges are assumed to be represented by the natural: [edge_type, batch_index, source_index, target_index].
			# Since attention is written from the perspective of the 'query' (i.e., the target), we swap the source and target columns to simplify computation.
			attention_bias = tf.stack([edge_ids[:, 0], edge_ids[:, 1], edge_ids[:, 3], edge_ids[:, 2]], axis=1)
		
		# If the input is 2-D, assume these are (batched) vocabulary indices and embed; otherwise, assume inputs are already embedded.
		if len(tf.shape(inputs)) == 2:
			states = self.embed_inputs(inputs)
		else:
			states = inputs
		
		# Finally, return the appropriate transformation.
		if is_enc_dec:
			return self.enc_dec_attention(inputs, masks, key_states, key_masks, attention_bias, training)
		else:
			return self.self_attention(inputs, masks, attention_bias, training)

	# Embed inputs. Note: applies scaling before positional encoding.
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
	def embed_inputs(self, inputs):
		states = tf.nn.embedding_lookup(self.embed, inputs)
		states *= tf.math.sqrt(tf.cast(tf.shape(states)[-1], "float32"))
		states += self.pos_enc[:tf.shape(states)[1]]
		return states
	
	# Standard self-attention, with dropout if training=True.
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, 4), dtype=tf.int32), tf.TensorSpec(shape=None, dtype=tf.bool)])
	def self_attention(self, states, masks, attention_bias, training):
		for ix in range(self.num_layers):
			new_states = (self.ln[ix][0](states),)
			new_states = self.attention[ix](new_states, masks, attention_bias)
			if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
			states += new_states
			
			new_states = self.ff_1[ix](self.ln[ix][1](states))
			if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
			new_states = self.ff_2[ix](new_states)
			if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
			states += new_states
		return self.ln_out(states)
	
	# Standard encoder-decoder attention, with dropout if training=True.
	# NOTE: tentatively does not support attention bias from the query domain to the key domain; extending this should be straightforward.
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, 4), dtype=tf.int32), tf.TensorSpec(shape=None, dtype=tf.bool)])
	def enc_dec_attention(self, states, masks, key_states, key_masks, attention_bias, training):
		for ix in range(self.num_layers):
			new_states = (self.ln[ix][0](states),)
			new_states = self.attention[ix](new_states, masks, tf.zeros((0,4), dtype='int32'))
			if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
			states += new_states

			new_states = self.ln[ix][2](states)
			new_states = self.enc_attention[ix]((new_states, key_states), key_masks, attention_bias)
			if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
			states += new_states
			
			new_states = self.ff_1[ix](self.ln[ix][1](states))
			if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
			new_states = self.ff_2[ix](new_states)
			if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
			states += new_states
		return self.ln_out(states)
	
	# Generates tokens from transformer states using the transposed embedding layer.
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)])
	def predict(self, states):
		return tf.matmul(states, self.embed, transpose_b=True)
	
	# Convenience function: returns a sequence mask in which each token can only see states up to its own position. Useful for generative language modeling (e.g. decoding).
	def get_sequence_mask(self, seq_len):
		return tf.sequence_mask(lengths=tf.range(1, seq_len + 1), maxlen=seq_len, dtype=tf.float32)
