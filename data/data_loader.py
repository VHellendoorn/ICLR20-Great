import os
import random

import json

import tensorflow as tf

class DataLoader():
	def __init__(self, data_path, data_config, vocabulary):
		self.data_path = data_path
		self.config = data_config
		self.vocabulary = vocabulary
	
	def batcher(self, mode="train"):
		data_path = self.get_data_path(mode)
		file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
		if mode == "train":
			random.shuffle(file_paths)
		ds = tf.data.Dataset.from_generator(self.to_batch, output_types=(tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32), args=(file_paths,))
		if mode == "dev": ds = ds.take(self.config['max_valid_samples'])
		ds = ds.prefetch(1)
		return ds
	
	def get_data_path(self, mode):	
		if mode == "train":
			return os.path.join(self.data_path, "train")
		elif mode == "dev":
			return os.path.join(self.data_path, "dev")
		elif mode == "eval":
			return os.path.join(self.data_path, "eval")
		else:
			raise ValueError("Mode % not supported for batching; please use \"train\", \"dev\", or \"eval\".")

	def to_sample(self, json_data):
		def parse_edges(edges):
			# Reorder edges to [edge type, source, target] and double edge type index to allow reverse edges
			relations = [[2*rel[2], rel[0], rel[1]] for rel in edges]
			relations += [[rel[0] + 1, rel[2], rel[1]] for rel in relations]  # Add reverse edges
			return relations
		
		tokens = [self.vocabulary.translate(t)[:self.config["max_token_length"]] for t in json_data["source_tokens"]]
		edges = parse_edges(json_data["edges"])
		error_location = json_data["error_location"]
		repair_targets = json_data["repair_targets"]
		repair_candidates = [t for t in json_data["repair_candidates"] if isinstance(t, int)]
		return (tokens, edges, error_location, repair_targets, repair_candidates)

	def to_batch(self, file_paths):
		def sample_len(sample):
			return len(sample[0])
		
		def make_batch(buffer):
			pivot = sample_len(random.choice(buffer))
			buffer = sorted(buffer, key=lambda b: abs(sample_len(b) - pivot))
			batch = []
			max_seq_len = 0
			for sample in buffer:
				max_seq_len = max(max_seq_len, sample_len(sample))
				if max_seq_len*(len(batch) + 1) > self.config['max_batch_size']:
					break
				batch.append(sample)
			batch_dim = len(batch)
			buffer = buffer[batch_dim:]
			batch = list(zip(*batch))
			
			# Pad all tokens to max length using ragged tensors
			token_tensor = tf.ragged.constant(batch[0], dtype=tf.dtypes.int32).to_tensor(shape=(len(batch[0]), max(len(b) for b in batch[0]), self.config["max_token_length"]))
			
			# Add batch axis to all edges and flatten
			edge_batches = tf.repeat(tf.range(batch_dim), [len(edges) for edges in batch[1]])
			edge_tensor = tf.concat(batch[1], axis=0)
			edge_tensor = tf.stack([edge_tensor[:, 0], edge_batches, edge_tensor[:, 1], edge_tensor[:, 0]], axis=1)

			# Error location is just a simple constant list
			error_location = tf.constant(batch[2], dtype=tf.dtypes.int32)
			
			# Targets and candidates both have an added batch dimension as well, and are otherwise just a list of indices
			target_batches = tf.repeat(tf.range(batch_dim), [len(targets) for targets in batch[3]])
			repair_targets = tf.cast(tf.concat(batch[3], axis=0), 'int32')  # Explicitly cast to int32 since repair targets can be empty, in which case TF defaults to float32
			repair_targets = tf.stack([target_batches, repair_targets], axis=1)
			
			candidates_batches = tf.repeat(tf.range(batch_dim), [len(candidates) for candidates in batch[4]])
			repair_candidates = tf.concat(batch[4], axis=0)
			repair_candidates = tf.stack([candidates_batches, repair_candidates], axis=1)
			
			return buffer, (token_tensor, edge_tensor, error_location, repair_targets, repair_candidates)
	
		buffer = []
		for file_path in file_paths:
			with open(file_path) as f:
				json_data = f.read()
			data = json.loads(json_data)
			for d in data:
				sample = self.to_sample(d)
				if sample_len(sample) > self.config['max_sequence_length']:
					continue
				buffer.append(sample)
				if sum(sample_len(sample) for l in buffer) > self.config['max_buffer_size']*self.config['max_batch_size']:
					buffer, batch = make_batch(buffer)
					if not batch: continue
					yield batch
		while buffer:
			buffer, batch = make_batch(buffer)
			if not batch: break
			yield batch

def main():
	dl = DataLoader('../../great', {'max_valid_samples': 1000})
	for b in dl.batcher(mode="dev"):
		print(b)

if __name__ == '__main__':
	main()