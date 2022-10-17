import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import pdb

import dgl.function as fn
import dgl.ops.edge_softmax as edge_softmax

class HCHALayer(nn.Module):
    def __init__(self, 
                    input_vdim, 
                    input_edim, 
                    output_vdim, 
                    output_edim,
                    num_heads=1, 
                    feat_drop=0, 
                    activation=None,
                    printflag=False, 
                    outputdir=""):
        super(HCHALayer, self).__init__()
           
        self._input_vdim = input_vdim
        self._input_edim = input_edim
        self._output_vdim = output_vdim
        self._output_edim = output_edim
        self._num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        
        self.p = nn.Linear(input_vdim, output_vdim * num_heads, bias=False)
        self.lin_v = nn.Linear(input_vdim, num_heads * output_vdim, bias=False)
        self.attn_v = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_vdim)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_vdim)))
        
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.p.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_v, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        
    def attention(self, edges):
        e = edges.src['e'] + edges.dst['e']        
        return {'e': e}

    def forward(self, g, vfeat, efeat, DV2, invDE):
        with g.local_scope():
            vfeat = self.feat_drop(vfeat)
            efeat = self.feat_drop(efeat)
            
            vfeat_p = self.p(vfeat).view(-1, self._num_heads, self._output_vdim)
            efeat_p = self.p(efeat).view(-1, self._num_heads, self._output_vdim)
            e_v = (vfeat_p * self.attn_v).sum(dim=-1).unsqueeze(-1)
            e_e = (efeat_p * self.attn_e).sum(dim=-1).unsqueeze(-1)
            
            g['con'].dstdata.update({'e': e_v})
            g['con'].srcdata.update({'e': e_e})
            g.apply_edges(self.attention, etype='con')
            e = F.relu(g['con'].edata.pop('e'))
            edata = edge_softmax(g['con'], e)
            g['con'].edata['a'] = edata
            g['in'].edata['a'] = edata
            
            input_ft = self.lin_v(vfeat) * DV2.view(-1,1)
            g['in'].srcdata.update({'ft': input_ft.view(-1, self._num_heads, self._output_vdim)})
            g.update_all(fn.u_mul_e('ft', 'a', 'm'),
                        fn.sum('m', 'ft'), etype='in')
            efeat = g['in'].dstdata['ft']
            efeat = efeat.mean(1)
            
            zefeat = efeat * invDE.view(-1,1)
            g['con'].srcdata.update({'ft': zefeat})
            g.update_all(fn.u_mul_e('ft', 'a', 'm'),
                        fn.sum('m', 'ft'), etype='con')
            vfeat = g['con'].dstdata['ft']
            vfeat = vfeat.mean(1)
            vfeat = vfeat * DV2.view(-1,1)
            
            if self.activation:
                vfeat = self.activation(vfeat)
            
        return [vfeat, efeat]

class HCHA(nn.Module):
    def __init__(self, 
                 input_vdim,
                 input_edim,
                 hidden_dim, 
                 vertex_dim,
                 edge_dim,
                 num_layers=3, 
                 num_heads=4, 
                 feat_drop=0.6):
        super(HCHA, self).__init__()
        
        self.activation = F.relu
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
                       
        if num_layers == 1:
             self.layers.append(HCHALayer(input_vdim, input_edim, vertex_dim, edge_dim, 
                                         num_heads=1, feat_drop=0., activation=None))
        else:
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(HCHALayer(input_vdim, input_edim, hidden_dim, hidden_dim,
                                                num_heads=num_heads, feat_drop=0., activation=self.activation))
                elif i == (num_layers - 1):
                    self.layers.append(HCHALayer(hidden_dim, hidden_dim,vertex_dim, edge_dim,
                                                num_heads=1, feat_drop=feat_drop, activation=None))
                else:
                    self.layers.append(HCHALayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim,
                                                num_heads=num_heads, feat_drop=feat_drop, activation=self.activation))
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
            
    def forward(self, g, vfeat, efeat, DV2, invDE):
        for l in range(self.num_layers - 1):
            vfeat, efeat = self.layers[l](g, vfeat, efeat, DV2, invDE)
        vfeat, efeat = self.layers[-1](g, vfeat, efeat, DV2, invDE)
        return vfeat, efeat
    