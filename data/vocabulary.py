import os
import random

import json

import tensorflow as tf

class Vocabulary():
	def __init__(self, vocab_path):
		self.vocab_path = vocab_path
		self.load_vocab()
	
	def load_vocab(self):
		with open(self.vocab_path, encoding='utf-8') as f:
			subtokens = [l.rstrip() for l in f]
		self.i2w = {ix+1:w for ix, w in enumerate(subtokens)}
		self.i2w[0] = "<PAD>"
		self.w2i = {w:ix for ix, w in self.i2w.items()}
		self.vocab_dim = len(self.i2w)
		
		# Some data structures to split up sub-tokenization
		self.bpe_cache = {}
		self.bpe_lookup_dict = {}
		for token in self.w2i.keys():
			if token[:2] not in self.bpe_lookup_dict:
				self.bpe_lookup_dict[token[:2]] = set()
			self.bpe_lookup_dict[token[:2]].add(token)
	
	def translate(self, token, is_subtokenized=False):
		return self.lookup(token) if is_subtokenized else [self.lookup(t) for t in self.tokenize(token)]
	
	def lookup(self, token):
		return self.w2i[token] if token in self.w2i else self.w2i["<PAD>"]  # Ignore truly unknown tokens; only happens when specific characters were never seen in training data.
	
	def tokenize(self, token):
		token += "#"  # Add terminal symbol first
		tokens = []
		ix = 0
		if token in self.bpe_cache:
			return self.bpe_cache[token]
		while ix < len(token):
			if ix == len(token) - 2:
				tokens.append(token[ix:])
				break
			else:
				candidates = self.bpe_lookup_dict.get(token[ix:ix+2], [])
				if not candidates:
					top_candidate = token[ix]
				else:
					# Only sub-tokens that match the next characters and don't leave the end-of-word marker left by itself
					candidates = [t for t in candidates if t == token[ix:ix+len(t)] and not len(token) == ix + len(t) + 1]
					if not candidates: top_candidate = token[ix]
					else: top_candidate = max(candidates, key=lambda e: len(e))
				tokens.append(top_candidate)
				ix += len(top_candidate)
		self.bpe_cache[token] = tokens
		return tokens