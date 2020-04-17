import tensorflow as tf


class RNN(tf.keras.layers.Layer):
	def __init__(self, model_config, vocab_dim):
		self.hidden_dim = model_config["hidden_dim"]
		self.num_layers = model_config["num_layers"]
		self.dropout_rate = model_config["dropout_rate"]
		self.vocab_dim = vocab_dim
	
	def build(self, _):
		self.embed = tf.Variable(random_init([self.vocab_dim, self.embed_dim]), dtype=tf.float32)
		self.rnns_fwd = [tf.keras.layers.GRU(self.hidden_dim//2, return_sequences=True) for _ in range(self.num_layers)]
		self.rnns_bwd = [tf.keras.layers.GRU(self.hidden_dim//2, return_sequences=True, go_backwards=True) for _ in range(self.num_layers)]
	
	def call(self, inputs, training):
		# If the input is 2-D, assume they are (batched) vocabulary indices and embed; otherwise, assume embedded inputs are already provided.
		if len(tf.shape(inputs)) == 2:
			states = tf.nn.embedding_lookup(self.embed, inputs)
		else:
			states = inputs
		
		# Run states through all layers.
		for layer_no in self.num_layers:
			fwd = self.rnns_fwd[layer_no](states)
			bwd = self.rnns_bwd[layer_no](states)
			states = tf.concat([fwd, bwd], axis=-1)			
			if training: states = tf.nn.dropout(states, rate=self.dropout_rate)
		return states
	