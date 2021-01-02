import config
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/4
def simple_elementwise_apply(fn, packed_sequence):
    """applies a pointwise function fn to each element in packed_sequence"""
    return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)

class Seq(nn.Module):
	def __init__(self, vocab_size, param, device, weight_tying = False):
		super(Seq, self).__init__()
		self.device = device
		
		# Model parameters
		self.vocab_size = vocab_size
		self.hidden_dim = param['hidden_dim']
		self.embed_dim = param['embed_dim']
		self.n_layers = param['n_layers']
		self.dropout_rate = param['dropout']
		
		# Define model architecture
		self.embedding = nn.Embedding(self.vocab_size , self.embed_dim)
		self.lstm = nn.LSTM(
			input_size = self.embed_dim,
			hidden_size = self.hidden_dim,
			num_layers = self.n_layers,
			bias = True, 				# default
			batch_first = True,
			dropout = self.dropout_rate,
			bidirectional = False 		# default
		)
		self.dropout = nn.Dropout(self.dropout_rate)
		self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
		if weight_tying:
			assert self.hidden_dim == self.embed_dim
			self.fc.weight = self.embedding.weight

	def forward(self, x, h, c):
		# e: [batch, seq len, emb]
		# o: [batch, seq len, hidden dim]
		# (h, c): [n layers, batch, hidden dim]
		# p: [batch, seq len, vocab size]
		
		e = self.embedding(x)
		o, (h, c) = self.lstm(e, (h, c))
		p = self.dropout(self.fc(o))
		return p, h, c
