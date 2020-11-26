import torch
import torch.nn as nn
import torch.nn.functional as F
import config

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Seq(nn.Module):
	def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout_rate, device=dev):
		super(Seq, self).__init__()
		self.vocab_size = vocab_size
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		self.embedding = nn.Embedding(vocab_size, embed_dim)
		self.lstm = nn.LSTM(
			input_size = embed_dim,
			hidden_size = hidden_dim,
			num_layers = n_layers,
			bias = True, 				# default
			batch_first = True,
			dropout = dropout_rate,
			bidirectional = False 		# default
		)
		self.dropout = nn.Dropout(dropout_rate)
		self.fc = nn.Linear(hidden_dim, vocab_size)

	def forward(self, x, h, c):
		# x: [batch, seq len] # Just seq len?

		e = self.dropout(self.embedding(x))
		# e: [batch, seq len, emb]

		e = nn.utils.rnn.pack_padded_sequence(
			e, 
			torch.Tensor(config.BATCH_SIZE).fill_(config.SEQ_LEN), 
			batch_first=True
		)
		o, (h, c) = self.lstm(e, (h, c))
		# o: [batch, seq len, hidden dim], (h, c): [n layers, batch, hidden dim]
		o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)

		# [batch * seq len, hidden dim]
		o = o.reshape(-1, o.shape[2])
		p = self.fc(o)
		return p, h, c