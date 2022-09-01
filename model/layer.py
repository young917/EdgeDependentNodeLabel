import torch
import torch.nn as nn
import torch, os, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from model.Transformer import MAB, SAB, ISAB, PMA
        
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
                # x = F.dropout(x, self.dropout, training=self.training)
        return x
        
class ScorerTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5, num_heads=4, num_inds=4, ln=False):
        super(ScorerTransformer, self).__init__()
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        self.num_layers = n_layers
        self.dropout = dropout
        
        self.enc = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.enc.append(ISAB(input_dim, hidden_dim, num_heads, num_inds, ln=ln))
            else:
                self.enc.append(ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=ln))
            self.enc.append(nn.Dropout(dropout))
        self.enc.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, X):
        X = X.unsqueeze(0)
        for layer in self.enc:
            X = layer(X)
        X = X.squeeze(0)
        return X

class InnerProduct(nn.Module):
    def __init__(self, output_dim):
        super(InnerProduct, self).__init__()
        self.output_dim = output_dim
        self.predict_layer = nn.Linear(1, self.output_dim)
        
    def forward(self, V, E):
        batchsize = V.shape[0]
        ip = torch.bmm(V.view(batchsize,1,-1), E.view(batchsize,-1,1)).squeeze(-1)
        output = self.predict_layer(ip)
        return output

class Wrap_Embedding(torch.nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, *input):
        return super().forward(*input), torch.Tensor([0]).to(device)
    
'''
class multilayers(nn.Module):
    def __init__(self, model, inputs, n_layers):
        super(multilayers, self).__init__()
        self.layers = []
        self.model = model
        for i in range(n_layers):
            self.layers.append(self.model(*inputs))
    
    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        return self
        
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
            
    def forward(self, inputs, n_layers):
        first_layer, last_layer = True, False
        for i, layer in enumerate(self.layers):
            if i == n_layers-1:
                last_layer = True
                return layer(*inputs, first_layer, last_layer)
            else:
                inputs = layer(*inputs, first_layer, last_layer)
            first_layer = False
        print('error')
    
class FCwSM(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(FCwSM, self).__init__()
        self.hidden_dim, self.output_dim = hidden_dim, output_dim
        self.predict_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.activation = nn.Softmax(dim=1)

    def forward(self, input):
        out = self.activation(self.predict_layer(input))
#         out = self.predict_layer(input)
        return out

class FCwTSM(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(FCwTSM, self).__init__()
        self.hidden_dim, self.output_dim = hidden_dim, output_dim
        #predict_layers = []
        #for _ in range(output_dim):
        #    predict_layers.append(nn.Linear(self.hidden_dim, 1))
        #self.predict_layers = nn.ModuleList(predict_layers)
        self.predict_layer = nn.Linear(self.hidden_dim, output_dim)
        # self.sm1 = torch.nn.Softmax(dim=1, inplace=False)
        # self.sm2 = torch.nn.Softmax(dim=0, inplace=False)
        
    def forward(self, input):
        #for i, layer in enumerate(self.predict_layers):
            #out = layer(input)
            # if i == len(self.predict_layers) - 1:
            #    out = F.softmax(out, dim=0)
            # print("out:", out.shape)
            #outs.append(out)
        out = self.predict_layer(input)
        out = F.softmax(out, dim=1)
        
        # outs = torch.split(out, 1, dim=1)
        # o = outs[2].clone()[:]
        # a = F.softmax(o, dim=0)
        # print(a)
        # outs[2] = a.clone()
        # print(outs[2])
        # out = torch.cat(outs, dim=1)
        
        out = F.softmax(out, dim=0)
        # out[:,2]. = F.softmax(out[:,2], dim=0).clone()
        out = F.softmax(out, dim=1)
        
        return out

'''