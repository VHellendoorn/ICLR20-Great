import tensorflow as tf


class GGNN(tf.keras.layers.Layer):
	def __init__(self, model_config, shared_embedding=None, vocab_dim=None):
		super(GGNN, self).__init__()
		self.num_edge_types = model_config['num_edge_types']
		# The main GGNN configuration is provided as a list of 'time-steps', which describes how often each layer is repeated.
		# E.g., an 8-step GGNN with 4 distinct layers repeated 3 and 1 times alternatingly can represented as [3, 1, 3, 1]
		self.time_steps = model_config['time_steps']
		self.num_layers = len(self.time_steps)
		# The residuals index in the time-steps above offset by one (index 0 refers to the node embeddings).
		# They describe short-cuts formatted as receiving layer: [sending layer] entries, e.g., {1: [0], 3: [0, 1]}
		self.residuals = {str(k):v for k, v in model_config['residuals'].items()}  # Keys must be strings for TF checkpointing
		self.hidden_dim = model_config['hidden_dim']
		self.add_type_bias = model_config['add_type_bias']
		self.dropout_rate = model_config['dropout_rate']
		
		# Initialize embedding variable in constructor to allow reuse by other models
		if shared_embedding is not None:
			self.embed = shared_embedding
		elif vocab_dim is None:
			raise ValueError('Pass either a vocabulary dimension or an embedding Variable')
		else:
			random_init = tf.random_normal_initializer(stddev=self.hidden_dim ** -0.5)
			self.embed = tf.Variable(random_init([vocab_dim, self.hidden_dim]), dtype=tf.float32)
	
	def build(self, _):
		# Small util functions
		random_init = tf.random_normal_initializer(stddev=self.hidden_dim ** -0.5)
		def make_weight(name=None):
			return tf.Variable(random_init([self.hidden_dim, self.hidden_dim]), name=name)
		def make_bias(name=None):
			return tf.Variable(random_init([self.hidden_dim]), name=name)
		
		# Set up type-transforms and GRUs
		self.type_weights = [[make_weight('type-' + str(j) + '-' + str(i)) for i in range(self.num_edge_types)] for j in range(self.num_layers)]
		self.type_biases = [[make_bias('bias-' + str(j) + '-' + str(i)) for i in range(self.num_edge_types)] for j in range(self.num_layers)]
		self.rnns = [tf.keras.layers.GRUCell(self.hidden_dim) for _ in range(self.num_layers)]
		for ix, rnn in enumerate(self.rnns):
			# Initialize the GRUs input dimension based on whether any residuals will be passed in.
			if str(ix) in self.residuals:
				rnn.build(self.hidden_dim*(1 + len(self.residuals[str(ix)])))
			else:
				rnn.build(self.hidden_dim)
	
	# Assume 'inputs' is an embedded batched sequence, 'edge_ids' is a sparse list of indices formatted as: [edge_type, batch_index, source_index, target_index].
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, 4), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.bool)])
	def call(self, states, edge_ids, training):
		# Collect some basic details about the graphs in the batch.
		edge_type_ids = tf.dynamic_partition(edge_ids[:, 1:], edge_ids[:, 0], self.num_edge_types)
		message_sources = [type_ids[:, 0:2] for type_ids in edge_type_ids]
		message_targets = [tf.stack([type_ids[:, 0], type_ids[:, 2]], axis=1) for type_ids in edge_type_ids]
		
		# Initialize the node_states with embeddings; then, propagate through layers and number of time steps for each layer.
		layer_states = [states]
		for layer_no, steps in enumerate(self.time_steps):
			for step in range(steps):
				if str(layer_no) in self.residuals:
					residuals = [layer_states[ix] for ix in self.residuals[str(layer_no)]]
				else:
					residuals = None
				new_states = self.propagate(layer_states[-1], layer_no, edge_type_ids, message_sources, message_targets, residuals=residuals)
				if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
				# Add or overwrite states for this layer number, depending on the step.
				if step == 0:
					layer_states.append(new_states)
				else:
					layer_states[-1] = new_states
		# Return the final layer state.
		return layer_states[-1]
	
	def propagate(self, in_states, layer_no, edge_type_ids, message_sources, message_targets, residuals=None):
		# Collect messages across all edge types.
		messages = tf.zeros_like(in_states)
		for type_index in range(self.num_edge_types):
			type_ids = edge_type_ids[type_index]
			if tf.shape(type_ids)[0] == 0:
				continue
			# Retrieve source states and compute type-transformation.
			edge_source_states = tf.gather_nd(in_states, message_sources[type_index])
			type_messages = tf.matmul(edge_source_states, self.type_weights[layer_no][type_index])
			if self.add_type_bias:
				type_messages += self.type_biases[layer_no][type_index]
			messages = tf.tensor_scatter_nd_add(messages, message_targets[type_index], type_messages)
		
		# Concatenate residual messages, if applicable.
		if residuals is not None:
			messages = tf.concat(residuals + [messages], axis=2)
		
		# Run GRU for each node.
		new_states, _ = self.rnns[layer_no](messages, tf.expand_dims(in_states, 0))
		return new_states[0]
	
	# Embed inputs. Note, does not add positional encoding.
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
	def embed_inputs(self, inputs):
		states = tf.nn.embedding_lookup(self.embed, inputs)
		states *= tf.math.sqrt(tf.cast(tf.shape(states)[-1], 'float32'))
		return states
