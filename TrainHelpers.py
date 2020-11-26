import numpy as np
import torch

def prep_batches(dataset, batch_size, seq_len):
	num_batches = len(dataset) // batch_size
	inputs = dataset[:num_batches * batch_size]
	targets = torch.zeros_like(inputs)
	for i in range(0, len(inputs)):
		targets[i][:-1] = inputs[i][1:] # skip first token
	inputs = inputs.view((num_batches, -1, seq_len))
	targets = targets.view((num_batches, -1, seq_len))
	return inputs, targets


def one_hot_encode(idx, vocab_size):
	one_hot = np.zeros(vocab_size)
	one_hot[idx] = 1
	return one_hot


def one_hot_encode_seq(sequence, vocab_size):
	encoding = np.array([
		one_hot_encode(token, vocab_size) for token in sequence
	])
	return encoding


def one_hot_encode_batch(batch, vocab_size):
	encoding = torch.tensor([
		one_hot_encode_seq(sequence, vocab_size) for sequence in batch
	])
	return encoding