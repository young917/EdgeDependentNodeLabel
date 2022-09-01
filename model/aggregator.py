import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
from attention import Multihead

class MeanAggregator(nn.Module):
    def __init__(self, dim_vertex):
        super(MeanAggregator, self).__init__()
        self.cls = nn.Linear(dim_vertex, 1)
    
    def forward(self, embeddings):
        #pdb.set_trace()
        embedding = embeddings.mean(dim=0).squeeze()
        #return F.softmax(self.cls(embedding))
        return torch.sigmoid(self.cls(embedding)), embedding
    
class SetAggregator(nn.Module):
    def __init__(self, dim_vertex, num_heads, num_seeds=1):
        super(SetAggregator, self).__init__()
        self.S = nn.Parameter(torch.Tensor(num_seeds, dim_vertex))
        nn.init.xavier_uniform_(self.S)
        self.attention = Multihead(dim_vertex, dim_vertex, dim_vertex, num_heads)
        
        #layers = [dim_vertex,2]
        layers = [dim_vertex,1]
        Layers = []
        for i in range(len(layers)-1):
            Layers.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                Layers.append(nn.ReLU(True))
        self.cls = nn.Sequential(*Layers)
        
    def forward(self, X):
        #pdb.set_trace()
        embedding = self.attention(self.S.repeat(X.size(0), 1), X).mean(dim=0)
        #return F.softmax(self.cls(embedding))
        return torch.sigmoid(self.cls(embedding)), embedding

    
class SAGNNAggregator(nn.Module):
    def __init__(self, dim_vertex, num_heads):
        super(SAGNNAggregator, self).__init__()
        self.static_embed = nn.Linear(dim_vertex, dim_vertex)
        self.dynamic_embed = Multihead(dim_vertex, dim_vertex, dim_vertex, num_heads)
        self.cls = nn.Linear(dim_vertex, 1)
        
    def forward(self, X):
        #pdb.set_trace()
        s_embed = torch.tanh(self.static_embed(X))
        d_embed = self.dynamic_embed(X, X)
        outputs = torch.sigmoid(self.cls(torch.pow(s_embed - d_embed, 2)))
        return outputs.mean(), torch.pow(s_embed - d_embed, 2)

    
class MaxminAggregator(nn.Module):
    def __init__(self, dim_vertex):
        super(MaxminAggregator, self).__init__()
        self.cls = nn.Linear(dim_vertex, 1)
    def forward(self, X):
        max_val, _ = torch.max(X, dim=0)
        min_val, _ = torch.min(X, dim=0)
        #pdb.set_trace()
        return torch.sigmoid(self.cls(max_val - min_val)), max_val - min_val