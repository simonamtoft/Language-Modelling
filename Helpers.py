import numpy as np
import torch
import config

HIDDEN_DIM = config.PARAM['hidden_dim']
N_LAYERS = config.PARAM['n_layers']


def prep_batches(dataset, batch_size, sequence_length, print_every = None):
    num_batches = (len(dataset)-1) // sequence_length // batch_size
    inputs = dataset[:num_batches*batch_size*sequence_length].view(-1, sequence_length)
    targets = dataset[1:num_batches*batch_size*sequence_length + 1].view(-1, sequence_length)

    # We now have (batches, batchsize, sequence_length) for both inputs and targets
    # However, we wish to have (i, j) to have the next sequence be (i+1, j) such that states between
    # batches can be used.

    temp_tuple = ([None] * num_batches, [None] * num_batches)
    for batch in range(0, num_batches):
        temp_tuple[0][batch] = inputs[batch:num_batches*batch_size:num_batches].view(batch_size, sequence_length)
        temp_tuple[1][batch] = targets[batch:num_batches*batch_size:num_batches].view(batch_size, sequence_length)
        if print_every and batch % print_every == 0:
            print("Preparing batch {}/{}".format(batch + 1, num_batches))

    inputs, targets = temp_tuple
    return inputs, targets
