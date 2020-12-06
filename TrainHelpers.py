import numpy as np
import torch
from torch.utils.data import DataLoader

def prep_batches(dataset, batch_size, print_every = None, num_workers=4):
	data_iter = iter(DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=num_workers))
	num_batches = len(data_iter) # len(dataset) // batch_size
	inputs =  [None] * num_batches
	targets = [None] * num_batches

	for batch in range(0, num_batches):
		ids = next(data_iter)["ids"]
		inputs[batch] = ids #torch.stack(dataset[batch*batch_size : (batch+1)*batch_size])
		targets[batch] = ids.clone()

		for sequence in range(0, batch_size):
			targets[batch][sequence] = torch.roll(targets[batch][sequence], -1, 0)
		if print_every and batch % print_every == 0:
			print("Preparing batch {}/{}".format(batch + 1, num_batches))
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