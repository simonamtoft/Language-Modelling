import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq(nn.Module):
	def __init__(self, vocab_size, param, device):
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

	def forward(self, x, h, c):
		e = self.embedding(x)
		
		# Why is dropout on embedding?
		#e = self.dropout(e)
		# e: [batch, seq len, emb]

		o, (h, c) = self.lstm(e, (h, c))
		# o: [batch, seq len, hidden dim], (h, c): [n layers, batch, hidden dim]

		p = self.dropout(self.fc(o))
		# [batch, seq len, vocab size]

		return p, h, c #p.to(self.device), h.to(self.device), c.to(self.device)
