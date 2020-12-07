import numpy as np
import torch

def prep_batches(dataset, batch_size, print_every = None):
    num_batches = len(dataset) // batch_size
    inputs = [None] * num_batches
    targets = [None] * num_batches

    # We wish to have (num_batches, batch_size) of tensors. 
	# If (i, j) is a sequence, then (i+1, j) should be the next sequence of the dataset.

    for batch in range(0, num_batches):
        inputs[batch] = dataset[batch:num_batches*batch_size:num_batches]
        targets[batch] = [None] * batch_size
        for sequence in range(0, batch_size):
            targets[batch][sequence] = torch.zeros_like(inputs[batch][sequence])
            targets[batch][sequence][:-1] = inputs[batch][sequence][1:]
            targets[batch][sequence][-1] = inputs[batch][sequence][0]
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