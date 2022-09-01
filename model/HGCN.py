import torch
import torch.nn as nn
import torch, os, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class HGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, numlayer, E_F):
        """
        an l-layer HGCN for output V, E embeddings
        """
        super(HGCN, self).__init__()
        self.input_dim, self.hidden_dim, self.numlayer = input_dim, hidden_dim, numlayer
        module_list = []
        for i in range(numlayer):
            if i == 0:
                if E_F > 0:
                    module_list.append(HGCN_LAYER(self.input_dim, self.hidden_dim, E_F))
                else:
                    module_list.append(HGCN_LAYER(self.input_dim, self.hidden_dim))
            else:
                module_list.append(HGCN_LAYER(self.hidden_dim, self.hidden_dim))
        self.layers = nn.ModuleList(module_list)

    def forward(self, V, E, A, D_V, D_E, D_V_inv, D_E_inv):
        for i, layer in enumerate(self.layers):
            if i == 0:
                V, E = layer(V, E, A, D_V, D_E, D_V_inv, D_E_inv)
            else:
                V, E = layer(V, None, A, D_V, D_E, D_V_inv, D_E_inv)
        return V, E


class HGCN_LAYER(nn.Module):
    def __init__(self, hidden_dim, next_hidden_dim, edge_feature_dim=0):
        super(HGCN_LAYER, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation = nn.ReLU()
        cuda = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.W_E = Parameter(torch.empty(hidden_dim, hidden_dim, device=cuda, dtype=torch.float))
        self.B_E = Parameter(torch.empty(1, hidden_dim, device=cuda, dtype=torch.float))
        self.W_V = Parameter(torch.empty(hidden_dim + edge_feature_dim, next_hidden_dim, device=cuda, dtype=torch.float))
        self.B_V = Parameter(torch.empty(1, next_hidden_dim, device=cuda, dtype=torch.float))
        self.reset_parameters()

    def forward(self, V, E, A, D_V, D_E, D_V_inv, D_E_inv):
        norm_V = torch.matmul(D_E_inv, torch.matmul(A.t(), torch.matmul(D_V, V)))
        if E is None:
            E = self.activation(torch.matmul(norm_V, self.W_E) + self.B_E)
        else:
            # concatenate with edge features
            E = torch.cat((self.activation(torch.matmul(norm_V, self.W_E) + self.B_E), E), dim=1)
        norm_E = torch.matmul(D_V_inv, torch.matmul(A, torch.matmul(D_E, E)))
        V = self.activation(torch.matmul(norm_E, self.W_V) + self.B_V)
        
        return V, E
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_E)
        torch.nn.init.xavier_uniform_(self.B_E)
        torch.nn.init.xavier_uniform_(self.W_V)
        torch.nn.init.xavier_uniform_(self.B_V)
