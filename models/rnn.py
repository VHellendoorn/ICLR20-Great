import tensorflow as tf


class RNN(tf.keras.layers.Layer):
	def __init__(self, model_config, shared_embedding=None, vocab_dim=None):
		super(RNN, self).__init__()
		self.hidden_dim = model_config['hidden_dim']
		self.num_layers = model_config['num_layers']
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
		self.rnns_fwd = [tf.keras.layers.GRU(self.hidden_dim//2, return_sequences=True) for _ in range(self.num_layers)]
		self.rnns_bwd = [tf.keras.layers.GRU(self.hidden_dim//2, return_sequences=True, go_backwards=True) for _ in range(self.num_layers)]
	
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.bool)])
	def call(self, states, training):
		states = tf.ensure_shape(states, (None, None, self.hidden_dim))
		# Run states through all layers.
		real_dropout_rate = self.dropout_rate * tf.cast(training, 'float32')  # Easier for distributed training than an explicit conditional
		for layer_no in range(self.num_layers):
			fwd = self.rnns_fwd[layer_no](states)
			bwd = self.rnns_bwd[layer_no](states)
			states = tf.concat([fwd, bwd], axis=-1)			
			states = tf.nn.dropout(states, rate=real_dropout_rate)
		return states
