import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
import math
import os
# Based on SetTransforme
from model.Whatsnet import MAB, SAB, ISAB, PMA
   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WhatsnetIM(nn.Module):
    def __init__(self, 
                 input_dim,
                 num_classes,
                 dim_hidden=128,
                 num_layer=1,
                 att_type="pure",
                 num_heads=4,
                 num_inds=4,
                 ln=False,
                 weight_flag=False,
                 pe_ablation_flag=False
                ):
        super(WhatsnetIM, self).__init__()
        
        self.input_dim, self.num_classes, self.dim_hidden = input_dim, num_classes, dim_hidden
        self.num_layer = num_layer
        self.att_type = att_type
        self.re_pe_flag = False
        self.pe_ablation_flag = pe_ablation_flag
        self.weight_flag = weight_flag
        if "RE" in self.att_type: # relative positional encoding
            self.re_pe_flag = True
        if self.att_type == "OrderPE":
            self.pe = nn.Linear(weight_dim, input_vdim)
        
        self.predict_layer = nn.ModuleList()
        dimension = input_dim
        for _ in range(self.num_layer):
            if self.att_type in ["ITRE", "ShawRE"]:
                self.predict_layer.append(SAB(dimension, dim_hidden, num_heads, ln=ln, RE=self.att_type_v))
                dimension = dim_hidden
            elif self.att_type != "NoAtt":
                if self.pe_ablation_flag:
                    self.predict_layer.append(SAB(dimension, dim_hidden, num_heads, ln=ln))
                else:
                    self.predict_layer.append(ISAB(dimension, dim_hidden, num_heads, num_inds, ln=ln))
                dimension = dim_hidden
        self.predict_layer.append(nn.Linear(dim_hidden, num_classes))
    
    def message_func(self, edges):
        if self.re_pe_flag:
            return {'v': edges.src['feat'], 'vindex': edges.src['index'], 'weight': edges.data['weight'], 'label': edges.data['label'].long()}
        elif self.weight_flag: # weight represents positional information
            return {'v': edges.src['feat'], 'weight': edges.data['weight'], 'label': edges.data['label'].long()}
        else:
            return {'v': edges.src['feat'], 'label': edges.data['label'].long()}
    
    def reduce_func(self, nodes):
        # nodes.mailbox['v' or 'q'].shape = (num batch, hyperedge size, feature dim)
        L = nodes.mailbox['label']
        v = nodes.mailbox['v']
        if self.weight_flag:
            W = nodes.mailbox['weight']
        
        if self.re_pe_flag:
            # 1. reduce last dimension
            # 2. sort column
            Vindex = nodes.mailbox['vindex']
            hedgesize = W.shape[1]
            batchsize = W.shape[0]
            W = W[:,:,:hedgesize]
            assert torch.sum(torch.flatten(W[:,:,hedgesize:])) == 0
            
            argsort_idx = torch.argsort(Vindex,1)
            argsort_idx = torch.flatten(argsort_idx, 1)
            for batch_idx in range(batchsize):
                W[batch_idx,:,:] = W[batch_idx,:,argsort_idx[batch_idx]]
            
        # Attention
        if self.att_type == "OrderPE":
            v = v + self.pe(W)
        for i, layer in enumerate(self.predict_layer):
            if self.re_pe_flag:
                v = layer(v, W)
            else:
                v = layer(v)
        
        self.output.append(v.reshape(-1, self.num_classes))
        self.labels.append(L.reshape(-1))
        
        return {'o': torch.sum(v, dim=1)}
        
    def forward(self, g, vfeat, efeat, vindex=None):
        self.output = []
        self.labels = []
        
        with g.local_scope():
            g.srcnodes['node'].data['feat'] = vfeat
            if self.re_pe_flag:
                assert vindex != None
                g.srcnodes['node'].data['index'] = vindex[:g['in'].num_src_nodes()]
            g.update_all(self.message_func, self.reduce_func, etype='in')
        
        output = torch.cat(self.output, dim=0)
        labels = torch.cat(self.labels, dim=0)
        
        return output, labels