import torch
import torch.nn as nn
import torch, os, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, dropout=0.6):
        super(FC, self).__init__()
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        self.num_layers = n_layers
        self.dropout = dropout
        self.predict_layer = nn.ModuleList()
        
        if self.num_layers == 1:
            self.predict_layer.append(nn.Linear(self.input_dim, self.output_dim))
        else:
            for i in range(self.num_layers):
                if i == 0:
                    self.predict_layer.append(nn.Linear(self.input_dim, self.hidden_dim))
                else:
                    self.predict_layer.append(nn.Linear(self.hidden_dim, self.output_dim))
        
    def forward(self, x):
        for i, layer in enumerate(self.predict_layer):
            if i == (self.num_layers - 1):
                x = layer(x)
            else:
                x = F.tanh(layer(x))
        return x

class Wrap_Embedding(torch.nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, *input):
        return super().forward(*input), torch.Tensor([0]).to(device)
