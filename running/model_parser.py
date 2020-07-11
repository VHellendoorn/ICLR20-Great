import tensorflow as tf

from models import great_transformer, ggnn, rnn, util

class VarMisuseModel(tf.keras.layers.Layer):
	def __init__(self, config, vocab_dim):
		super(VarMisuseModel, self).__init__()
		self.config = config
		self.vocab_dim = vocab_dim
	
	def build(self, _):
		# These layers are always used; initialize with any given model's hidden_dim
		random_init = tf.random_normal_initializer(stddev=self.config['base']['hidden_dim'] ** -0.5)
		self.embed = tf.Variable(random_init([self.vocab_dim, self.config['base']['hidden_dim']]), dtype=tf.float32)
		self.prediction = tf.keras.layers.Dense(2) # Two pointers: bug location and repair
		
		# Store for convenience
		self.pos_enc = tf.constant(util.positional_encoding(self.config['base']['hidden_dim'], 5000))
		
		# Next, parse the main 'model' from the config
		join_dicts = lambda d1, d2: {**d1, **d2}  # Small util function to combine configs
		base_config = self.config['base']
		desc = self.config['configuration'].split(' ')
		self.stack = []
		for kind in desc:
			if kind == 'rnn':
				self.stack.append(rnn.RNN(join_dicts(self.config['rnn'], base_config), shared_embedding=self.embed))
			elif kind == 'ggnn':
				self.stack.append(ggnn.GGNN(join_dicts(self.config['ggnn'], base_config), shared_embedding=self.embed))
			elif kind == 'great':
				self.stack.append(great_transformer.Transformer(join_dicts(self.config['transformer'], base_config), shared_embedding=self.embed))
			elif kind == 'transformer':  # Same as above, but explicitly without bias_dim set -- defaults to regular Transformer.
				joint_config = join_dicts(self.config['transformer'], base_config)
				joint_config['num_edge_types'] = None
				self.stack.append(great_transformer.Transformer(joint_config, shared_embedding=self.embed))
			else:
				raise ValueError('Unknown model component provided:', kind)
	
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.int32), tf.TensorSpec(shape=(None, None), dtype=tf.int32), tf.TensorSpec(shape=(None, 4), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.bool)])
	def call(self, tokens, token_mask, edges, training):
		# Embed subtokens and average into token-level embeddings, masking out invalid locations
		subtoken_embeddings = tf.nn.embedding_lookup(self.embed, tokens)
		subtoken_embeddings *= tf.expand_dims(tf.cast(tf.clip_by_value(tokens, 0, 1), dtype='float32'), -1)
		states = tf.reduce_mean(subtoken_embeddings, 2)
		
		# Track whether any position-aware model processes the states first. If not, add positional encoding to ensure that e.g. GREAT and GGNN
		# have sequential awareness. This is especially (but not solely) important because the default, non-buggy 'location' is the 0th token,
		# which is hard to predict for e.g. Transformers and GGNNs without either sequential awareness or a special marker at that location.
		if not self.stack or not isinstance(self.stack[0], rnn.RNN):
			states += self.pos_enc[:tf.shape(states)[1]]
		
		# Pass states through all the models (may be empty) in the parsed stack.
		for model in self.stack:
			if isinstance(model, rnn.RNN):  # RNNs simply use the states
				states = model(states, training=training)
			elif isinstance(model, ggnn.GGNN):  # For GGNNs, pass edges as-is
				states = model(states, edges, training=training)
			elif isinstance(model, great_transformer.Transformer):  # For Transformers, reverse edge directions to match query-key direction and add attention mask.
				mask = tf.cast(token_mask, dtype='float32')
				mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)
				attention_bias = tf.stack([edges[:, 0], edges[:, 1], edges[:, 3], edges[:, 2]], axis=1)
				states = model(states, mask, attention_bias, training=training)  # Note that plain transformers will simply ignore the attention_bias.
			else:
				raise ValueError('Model not yet supported:', model)
		
		# Finally, predict a simple 2-pointer outcome from these states, and return
		predictions = self.prediction(states)
		predictions = tf.transpose(self.prediction(states), [0, 2, 1])  # Convert to [batch, 2, seq-length] for convenience.
		return predictions
	
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, 2, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None), dtype=tf.int32), tf.TensorSpec(shape=(None,), dtype=tf.int32), tf.TensorSpec(shape=(None, None), dtype=tf.int32), tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
	def get_loss(self, predictions, token_mask, error_locations, repair_targets, repair_candidates):
		# Mask out infeasible tokens in the logits
		seq_mask = tf.cast(token_mask, 'float32')
		predictions += (1.0 - tf.expand_dims(seq_mask, 1)) * tf.float32.min
		
		# Used to calculate metrics specifically for (non-)buggy samples
		is_buggy = tf.cast(tf.clip_by_value(error_locations, 0, 1), 'float32')  # 0 is the default position for non-buggy samples
		
		# Localization loss is simply calculated with sparse CE
		loc_predictions = predictions[:, 0]
		loc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(error_locations, loc_predictions)
		loc_loss = tf.reduce_mean(loc_loss)
		loc_accs = tf.keras.metrics.sparse_categorical_accuracy(error_locations, loc_predictions)
		
		# Store two metrics: the accuracy at predicting specifically the non-buggy samples correctly (to measure false alarm rate), and the accuracy at detecting the real bugs.
		no_bug_pred_acc = tf.reduce_sum((1 - is_buggy) * loc_accs) / (1e-9 + tf.reduce_sum(1 - is_buggy))  # Take mean only on sequences without errors
		bug_loc_acc = tf.reduce_sum(is_buggy * loc_accs) / (1e-9 + tf.reduce_sum(is_buggy))  # Only on errors
		
		# For repair accuracy, first convert to probabilities, masking out any non-candidate tokens
		pointer_logits = predictions[:, 1]
		candidate_mask = tf.scatter_nd(repair_candidates, tf.ones(tf.shape(repair_candidates)[0]), tf.shape(pointer_logits))
		pointer_logits += (1.0 - candidate_mask) * tf.float32.min
		pointer_probs = tf.nn.softmax(pointer_logits)
		
		# Aggregate probabilities at repair targets to get the sum total probability assigned to the correct variable name
		target_mask = tf.scatter_nd(repair_targets, tf.ones(tf.shape(repair_targets)[0]), tf.shape(pointer_probs))
		target_probs = tf.reduce_sum(target_mask * pointer_probs, -1)
		
		# The loss is only computed at buggy samples, using (negative) cross-entropy
		target_loss = tf.reduce_sum(is_buggy * -tf.math.log(target_probs + 1e-9)) / (1e-9 + tf.reduce_sum(is_buggy))  # Only on errors
		
		# To simplify the comparison, accuracy is computed as achieving >= 50% probability for the top guess
		# (as opposed to the slightly more accurate, but hard to compute quickly, greatest probability among distinct variable names).
		rep_accs = tf.cast(tf.greater_equal(target_probs, 0.5), 'float32')
		target_loc_acc = tf.reduce_sum(is_buggy * rep_accs) / (1e-9 + tf.reduce_sum(is_buggy))  # Only on errors
		
		# Also store the joint localization and repair accuracy -- arguably the most important metric.
		joint_acc = tf.reduce_sum(is_buggy * loc_accs * rep_accs) / (1e-9 + tf.reduce_sum(is_buggy))  # Only on errors
		return (loc_loss, target_loss), (no_bug_pred_acc, bug_loc_acc, target_loc_acc, joint_acc)
