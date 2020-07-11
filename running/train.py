import sys
sys.path.append('.')

import argparse
import yaml

import numpy as np
import tensorflow as tf

from checkpoint_tracker import Tracker
from data import data_loader, vocabulary
from model_parser import VarMisuseModel

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("data_path", help="Path to data root")
	ap.add_argument("vocabulary_path", help="Path to vocabulary file")
	ap.add_argument("config", help="Path to config file")
	ap.add_argument("-m", "--models", help="Directory to store trained models (optional)")
	ap.add_argument("-l", "--log", help="Path to store training log (optional)")
	args = ap.parse_args()
	config = yaml.safe_load(open(args.config))
	print("Training with configuration:", config)
	data = data_loader.DataLoader(args.data_path, config["data"], vocabulary.Vocabulary(args.vocabulary_path))
	train(data, config, args.models, args.log)

def train(data, config, model_path=None, log_path=None):
	model = VarMisuseModel(config, data.vocabulary.vocab_dim)
	optimizer = tf.optimizers.Adam(config["training"]["learning_rate"])

	# Restore model from checkpoints if present; also sets up logger
	if model_path is None:
		tracker = Tracker(model)
	else:
		tracker = Tracker(model, model_path, log_path)

	first = True
	mbs = 0
	losses = [tf.keras.metrics.Mean() for _ in range(2)]
	accs = [tf.keras.metrics.Mean() for _ in range(4)]
	counts = [tf.keras.metrics.Sum(dtype='int32') for _ in range(2)]
	while tracker.ckpt.step < config["training"]["max_steps"]:
		if not first:
			print("Step:", tracker.ckpt.step.numpy() + 1)
		# These are just for console logging, not global counts
		for batch in data.batcher(mode='train'):
			mbs += 1
			tokens, edges, error_loc, repair_targets, repair_candidates = batch
			token_mask = tf.clip_by_value(tf.reduce_sum(tokens, -1), 0, 1)
			
			# Track the first batch to allow proper restoration of models (i.e., after variables have been init'd) in Eager mode.
			if first:
				model(tokens, token_mask, edges, training=False) # Run the first batch to allow proper restore of parameters
				print("Model initialized, training {:,} parameters".format(np.sum([np.prod(v.shape) for v in model.trainable_variables])))
				tracker.restore()
				if tracker.ckpt.step.numpy() > 0:
					print("Restored from step:", tracker.ckpt.step.numpy() + 1)
				else:
					print("Step:", tracker.ckpt.step.numpy() + 1)
				first = False
			
			with tf.GradientTape() as tape:
				pointer_preds = model(tokens, token_mask, edges, training=True)
				ls, acs = model.get_loss(pointer_preds, token_mask, error_loc, repair_targets, repair_candidates)
				loc_loss, rep_loss = ls
				loss = loc_loss + rep_loss

			grads = tape.gradient(loss, model.trainable_variables)
			grads, _ = tf.clip_by_global_norm(grads, 0.25)
			optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

			# Update statistics
			num_buggy = tf.reduce_sum(tf.clip_by_value(error_loc, 0, 1))
			samples = tf.shape(token_mask)[0]
			prev_samples = tracker.get_samples()
			curr_samples = tracker.update_samples(samples)
			update_metrics(losses, accs, counts, token_mask, ls, acs, num_buggy)
		
			# Every few minibatches, print the recent training performance
			if mbs % config["training"]["print_freq"] == 0:
				avg_losses = ["{0:.3f}".format(l.result().numpy()) for l in losses]
				avg_accs = ["{0:.2%}".format(a.result().numpy()) for a in accs]
				print("MB: {0}, seqs: {1:,}, tokens: {2:,}, loss: {3}, accs: {4}".format(mbs, curr_samples, counts[1].result().numpy(), ", ".join(avg_losses), ", ".join(avg_accs)))
				[l.reset_states() for l in losses]
				[a.reset_states() for a in accs]
			
			# Every valid_interval samples, run an evaluation pass and store the most recent model with its heldout accuracy
			if prev_samples // config["data"]["valid_interval"] < curr_samples // config["data"]["valid_interval"]:
				avg_accs = evaluate(data, config, model)
				tracker.save_checkpoint(model, avg_accs)
				if tracker.ckpt.step >= config["training"]["max_steps"]:
					break
				else:
					print("Step:", tracker.ckpt.step.numpy() + 1)

def evaluate(data, config, model):  # Similar to train, just without gradient updates
	print("Running evaluation pass on heldout data")
	losses = [tf.keras.metrics.Mean() for _ in range(2)]
	accs = [tf.keras.metrics.Mean() for _ in range(4)]
	counts = [tf.keras.metrics.Sum(dtype='int32') for _ in range(2)]
	for batch in data.batcher(mode='dev'):
		tokens, edges, error_loc, repair_targets, repair_candidates = batch		
		token_mask = tf.clip_by_value(tf.reduce_sum(tokens, -1), 0, 1)
		
		pointer_preds = model(tokens, token_mask, edges, training=True)
		ls, acs = model.get_loss(pointer_preds, token_mask, error_loc, repair_targets, repair_candidates)
		num_buggy = tf.reduce_sum(tf.clip_by_value(error_loc, 0, 1))
		update_metrics(losses, accs, counts, token_mask, ls, acs, num_buggy)
		if counts[0].result() > config['data']['max_valid_samples']:
			break
	avg_accs = [a.result().numpy() for a in accs]
	avg_accs_str = ", ".join(["{0:.2%}".format(a) for a in avg_accs])
	avg_loss_str = ", ".join(["{0:.3f}".format(l.result().numpy()) for l in losses])
	print("Evaluation result: seqs: {0:,}, tokens: {1:,}, loss: {2}, accs: {3}".format(counts[0].result().numpy(), counts[1].result().numpy(), avg_loss_str, avg_accs_str))
	return avg_accs

def get_metrics():
	losses = [tf.keras.metrics.Mean() for _ in range(2)]
	accs = [tf.keras.metrics.Mean() for _ in range(4)]
	counts = [tf.keras.metrics.Sum(dtype='int32') for _ in range(2)]
	return loss, accs, counts

def update_metrics(losses, accs, counts, token_mask, ls, acs, num_buggy_samples):
	loc_loss, rep_loss = ls
	no_bug_pred_acc, bug_loc_acc, target_loc_acc, joint_acc = acs
	num_samples = tf.shape(token_mask)[0]
	counts[0].update_state(num_samples)
	counts[1].update_state(tf.reduce_sum(token_mask))
	losses[0].update_state(loc_loss)
	losses[1].update_state(rep_loss)
	accs[0].update_state(no_bug_pred_acc, sample_weight=num_samples - num_buggy_samples)
	accs[1].update_state(bug_loc_acc, sample_weight=num_buggy_samples)
	accs[2].update_state(target_loc_acc, sample_weight=num_buggy_samples)
	accs[3].update_state(joint_acc, sample_weight=num_buggy_samples)

if __name__ == '__main__':
	main()