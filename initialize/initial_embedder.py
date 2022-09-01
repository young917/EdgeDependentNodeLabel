import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm, trange
import copy
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SparseEmbedding(nn.Module):
    def __init__(self, embedding_weight, sparse=True):
        super().__init__()
        print(embedding_weight.shape)
        self.sparse = sparse
        if self.sparse:
            self.embedding = embedding_weight
        else:
            try:
                try:
                    self.embedding = torch.from_numpy(
                        np.asarray(embedding_weight.todense())).to(device)
                except BaseException:
                    self.embedding = torch.from_numpy(
                        np.asarray(embedding_weight)).to(device)
            except Exception as e:
                print("Sparse Embedding Error",e)
                self.sparse = True
                self.embedding = embedding_weight
    
    def forward(self, x):
        
        if self.sparse:
            x = x.cpu().numpy()
            x = x.reshape((-1))
            temp = np.asarray((self.embedding[x, :]).todense())
            
            return torch.from_numpy(temp).to(device)
        else:
            return self.embedding[x, :]


class TiedAutoEncoder(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.Tensor(out, inp))
        self.bias1 = nn.parameter.Parameter(torch.Tensor(out))
        self.bias2 = nn.parameter.Parameter(torch.Tensor(inp))
        
        self.register_parameter('tied weight',self.weight)
        self.register_parameter('tied bias1', self.bias1)
        self.register_parameter('tied bias2', self.bias2)
        
        self.reset_parameters()
        
    

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias1 is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias1, -bound, bound)
        
        if self.bias2 is not None:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            torch.nn.init.uniform_(self.bias2, -bound, bound)

    def forward(self, input):
        encoded_feats = F.linear(input, self.weight, self.bias1)
        encoded_feats = F.tanh(encoded_feats)
        reconstructed_output = F.linear(encoded_feats, self.weight.t(), self.bias2)
        return encoded_feats, reconstructed_output
    
class MultipleEmbedding(nn.Module):
    def __init__(
            self,
            embedding_weights,
            dim,
            sparse=True,
            num=None):
        super().__init__()
        print(dim)
        self.num = num
        self.dim = dim
        
        test = torch.zeros(1, device=device).long()
        self.embeddings = SparseEmbedding(embedding_weights, sparse)
        self.input_size = self.embeddings(test).shape[-1]

        self.w = TiedAutoEncoder(self.input_size,self.dim).to(device)
        self.norm =nn.LayerNorm(self.dim).to(device)
        
        self.add_module("Embedding_Linear", self.w)
        self.add_module("Embedding_norm", self.norm)
            
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # maybe #batch 1 dimension long tensor
        sz_b = x.shape[0]
        recon_loss = torch.Tensor([0.0]).to(device)
        adj = self.embeddings(x)
        output = self.dropout(adj)
        output, recon = self.w(output)
        output = self.norm(output)
        recon_loss += sparse_autoencoder_error(recon, adj)
        
        return output, recon_loss

def sparse_autoencoder_error(y_pred, y_true):
    return torch.mean(torch.sum((y_true.ne(0).type(torch.float) * (y_true - y_pred)) ** 2, dim = -1) / torch.sum(y_true.ne(0).type(torch.float), dim = -1))
