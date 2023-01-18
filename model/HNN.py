import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import pdb

import dgl.function as fn
import dgl.ops.edge_softmax as edge_softmax

class HNNLayer(nn.Module):
    def __init__(self, 
                    input_vdim, 
                    input_edim, 
                    output_vdim, 
                    output_edim,
                    feat_drop=0, 
                    num_psi_layer=1,
                    activation=None,
                    printflag=False, 
                    outputdir=""):
        super(HNNLayer, self).__init__()
           
        self._input_vdim = input_vdim
        self._input_edim = input_edim
        self._output_vdim = output_vdim
        self._output_edim = output_edim
        self.num_psi_layer = num_psi_layer
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        
        self.weight_v = nn.Linear(input_vdim, output_vdim, bias=False)
        self.weight_e = nn.Linear(input_edim, output_edim, bias=False)
        
        self.psi_1 = nn.ModuleList()
        self.psi_2 = nn.ModuleList()
        dim1 = input_vdim + input_edim
        dim2 = output_vdim + input_edim
        for _ in range(self.num_psi_layer):
            self.psi_1.append(nn.Linear(dim1, input_vdim))
            self.psi_2.append(nn.Linear(dim2, input_edim))
            dim1 = input_vdim
            dim2 = input_edim
        
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.weight_v.weight, gain=gain)
        nn.init.xavier_normal_(self.weight_e.weight, gain=gain)
        for layer1, layer2 in zip(self.psi_1, self.psi_2):
#             print("psi weight", layer1.weight.shape)
            nn.init.xavier_normal_(layer1.weight, gain=gain)
            nn.init.xavier_normal_(layer2.weight, gain=gain)
        
    def v_message_psi_func(self, edges):
        return {'v': edges.src['feat'], 'e': edges.dst['feat'], 'weight': edges.src['dv']}
    
    def v_reduce_psi1(self, nodes):
        E = nodes.mailbox['e']
        V = nodes.mailbox['v']
        W = nodes.mailbox['weight']
#         print("V", V.shape)
#         print("E", E.shape)
        v = torch.cat([V, E], dim=2)
#         print("v", v.shape)
        for psi_layer in self.psi_1:
#             print(psi_layer.weight.shape)
            v = psi_layer(v)
#         print("psi1 v", v.shape, "psi1 w", W.shape)
#         print(v[0][0])
#         print(W[0])
        v = v * W.unsqueeze(-1)
#         print(v[0][0])
        o = torch.sum(v, dim=1)
        return {'o': o}
    
    def v_reduce_psi2(self, nodes):
        E = nodes.mailbox['e']
        V = nodes.mailbox['v']
        W = nodes.mailbox['weight']
        
        v = torch.cat([V, E], dim=2)
        for psi_layer in self.psi_2:
            v = psi_layer(v)
        o = torch.sum(v, dim=1)
        return {'o': o}
    
    def message_func(self, edges):
        return {'ft': edges.src['feat']}
    
    def reduce_func(self, nodes):
        feat = nodes.mailbox['ft']
        o = torch.sum(feat, dim=1)
        return {'o': o}
    
    def forward(self, g, vfeat, efeat, invDV, invDE, vMat, eMat):
        with g.local_scope():
            vfeat = self.feat_drop(vfeat)
            efeat = self.feat_drop(efeat)
            
            g['in'].srcdata.update({'dv': invDV, 'feat': vfeat})
            g['in'].dstdata.update({'feat': efeat})
            g['in'].update_all(self.v_message_psi_func, self.v_reduce_psi1, etype='in')
            A = g['in'].dstdata['o']
#             print ("A", A.shape)
#             print(A[0])
            A = torch.sparse.mm(eMat, A)
            _efeat = A + efeat
            g['con'].srcdata.update({'feat': _efeat})
            g['con'].update_all(self.message_func, self.reduce_func, etype='con')
            _vfeat = g['con'].dstdata['o']
            _vfeat = self.weight_v(_vfeat)
            vfeat = self.activation(_vfeat)
            
            g['in'].srcdata.update({'dv': torch.ones_like(invDV), 'feat': vfeat})
            g['in'].dstdata.update({'feat': efeat})
            g['in'].update_all(self.v_message_psi_func, self.v_reduce_psi2, etype='in')
            B = g['in'].dstdata['o']
            B = B * invDE.view(-1,1)
            g['con'].srcdata.update({'feat' : efeat})
            g['con'].update_all(self.message_func, self.reduce_func, etype='con')
            _vfeat = g['con'].dstdata['o']
            _vfeat = torch.sparse.mm(vMat, _vfeat)
            g['in'].srcdata.update({'dv': torch.ones_like(invDV), 'feat': _vfeat})
            g['in'].update_all(self.message_func, self.reduce_func, etype='in')
            _efeat = g['in'].dstdata['o']
            _efeat = _efeat + B
            _efeat = self.weight_e(_efeat)
            efeat = self.activation(_efeat)
            
        return [vfeat, efeat]

class HNN(nn.Module):
    def __init__(self, 
                 input_vdim,
                 input_edim,
                 hidden_dim, 
                 vertex_dim,
                 edge_dim,
                 num_psi_layer=1,
                 num_layers=3,
                 feat_drop=0.6):
        super(HNN, self).__init__()
        
        self.activation = F.relu
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
                       
        if num_layers == 1:
             self.layers.append(HNNLayer(input_vdim, input_edim, vertex_dim, edge_dim, num_psi_layer=num_psi_layer, feat_drop=0., activation=self.activation))
        else:
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(HNNLayer(input_vdim, input_edim, hidden_dim, hidden_dim, num_psi_layer=num_psi_layer, feat_drop=0., activation=self.activation))
                elif i == (num_layers - 1):
                    self.layers.append(HNNLayer(hidden_dim, hidden_dim,vertex_dim, edge_dim, num_psi_layer=num_psi_layer, feat_drop=feat_drop, activation=self.activation))
                else:
                    self.layers.append(HNNLayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_psi_layer=num_psi_layer, feat_drop=feat_drop, activation=self.activation))
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
            
    def forward(self, g, vfeat, efeat, invDV, invDE, vMat, eMat):
        for l in range(self.num_layers - 1):
            vfeat, efeat = self.layers[l](g, vfeat, efeat, invDV, invDE, vMat, eMat)
        vfeat, efeat = self.layers[-1](g, vfeat, efeat, invDV, invDE, vMat, eMat)
        return vfeat, efeat
    