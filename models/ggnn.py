import tensorflow as tf


class GGNN(tf.keras.layers.Layer):
	def __init__(self, model_config, vocab_dim, num_edge_types):
		super(GraphModel, self).__init__()
		self.vocab_dim = vocab_dim
		self.num_edge_types = num_edge_types
		
		# The main GGNN configuration is provided as a list of "time-steps", which describes how often each layer is repeated.
		# E.g., an 8-step GGNN with 4 distinct layers repeated 3 and 1 times alternatingly can represented as [3, 1, 3, 1]
		self.time_steps = model_config["time_steps"]
		self.num_layers = len(self.time_steps)
		# The residuals index in the time-steps above offset by one (index 0 refers to the node embeddings).
		# They describe short-cuts formatted as receiving layer: [sending layer] entries, e.g., {1: [0], 3: [0, 1]}
		self.residuals = model_config["residuals"]
		self.embed_dim = model_config["embed_dim"]
		self.hidden_dim = model_config["hidden_dim"]
		self.add_type_bias = model_config["add_type_bias"]
		self.dropout_rate = model_config["dropout_rate"]
	
	def build(self, _):
		# Small util functions
		random_init = tf.random_normal_initializer(stddev=self.hidden_dim ** -0.5)
		def make_weight(name=None):
			return tf.Variable(random_init([self.hidden_dim, self.hidden_dim]), name=name)
		def make_bias(name=None):
			return tf.Variable(random_init([self.hidden_dim]), name=name)
		
		# Set up embedding, type-transform and GRUs
		self.embed = tf.Variable(random_init([self.vocab_dim, self.embed_dim]), dtype=tf.float32)
		self.type_weights = [[make_weight("type-" + str(j) + "-" + str(i)) for i in range(self.num_edge_types)] for j in range(self.num_layers)]
		self.type_biases = [[make_bias("bias-" + str(j) + "-" + str(i)) for i in range(self.num_edge_types)] for j in range(self.num_layers)]
		self.rnns = [tf.keras.layers.GRUCell(self.hidden_dim) for _ in range(self.num_layers)]
		for ix, rnn in enumerate(self.rnns):
			# Initialize the GRUs input dimension based on whether any residuals will be passed in.
			if ix in self.residuals:
				rnn.build(self.hidden_dim*(1 + len(self.residuals[ix])))
			else:
				rnn.build(self.hidden_dim)
	
	# Assume 'inputs' is a batched sequence, 'edge_ids' is a sparse list of indices formatted as: [edge_type, batch_index, source_index, target_index].
	def call(self, inputs, edge_ids, training):
		# Collect some basic details about the graphs in the batch.
		edge_type_ids = tf.dynamic_partition(edge_ids[:, 0], edge_ids[:, 1:], self.num_edge_types)
		message_sources = [type_ids[:, 1:3] for type_ids in edge_type_ids]
		message_targets = [tf.stack([type_ids[:, 1], type_ids[:, 3]], axis=1) for type_ids in edge_type_ids]
		
		# If the input is 2-D, assume they are (batched) vocabulary indices and embed; otherwise, assume embedded inputs are already provided.
		if len(tf.shape(inputs)) == 2:
			embeddings = tf.nn.embedding_lookup(self.embed, inputs)
		else:
			embeddings = inputs
		
		# Initialize the node_states with embeddings; then, propagate through layers and number of time steps for each layer.
		layer_states = [embeddings]
		for layer_no, steps in enumerate(self.time_steps):
			for step in range(steps):
				if layer_no in self.residuals:
					residual = [layer_states[ix] for ix in self.residuals[layer_no]]
				else:
					residual = None
				new_states = self.propagate(layer_states[-1], layer_no, edge_type_ids, message_sources, message_targets, residual=residual)
				if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
				# Add or overwrite states for this layer number, depending on the step.
				if step == 0:
					layer_states.append(new_states)
				else:
					layer_states[-1] = new_states
		# Return the final layer state.
		return layer_states[-1]
	
	def propagate(self, in_states, layer_no, edge_type_ids, message_sources, message_targets, residual=None):
		# Collect messages across all edge types.
		messages = tf.zeros_like(in_states)
		for type_index in self.num_edge_types:
			type_ids = edge_type_ids[type_index]
			if tf.shape(type_ids)[0] == 0:
				continue
			# Retrieve source states and compute type-transformation.
			edge_source_states = tf.gather_nd(in_states, message_sources)
			type_messages = tf.matmul(edge_source_states, self.type_weights[layer_no][type_index])
			if self.add_type_bias:
				type_messages += self.type_biases[layer_no][type_index]
			messages = tf.tensor_scatter_nd_add(messages, message_targets, type_messages)
		
		# Concatenate residual messages, if applicable.
		if residual is not None:
			messages = tf.concat(residual + [messages], axis=1)
		
		# Run GRU for each node.
		new_states, _ = self.rnns[layer_no](messages, tf.expand_dims(in_states, 0))
		return new_states
