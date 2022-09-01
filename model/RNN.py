import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os

class ScorerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5, bidir_flag=True):
        super(ScorerGRU, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.n_layers = n_layers
        
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, bidirectional=True, num_layers=self.n_layers, batch_first=True, dropout=self.dropout)
        self.predict_layer = nn.Linear(2 * self.hidden_dim, self.output_dim)
        self.sm = nn.Softmax(dim=1)
    
    def initiate_hidden(self, batchsize, device):
        self.hidden_cell = torch.zeros(2 * self.n_layers, 1, self.hidden_dim).to(device)
    
    def forward(self, input_seq):
        # input_seq = (size of hyperedge, embedding dimension)
        out, hidden = self.gru(input_seq.view(1, input_seq.shape[0], -1), self.hidden_cell)
        predictions = self.sm(self.predict_layer(out.view(input_seq.shape[0], -1)))
        
        return predictions
        