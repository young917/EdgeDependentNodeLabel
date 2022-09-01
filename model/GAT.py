import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import pdb

import dgl.function as fn
import dgl.ops.edge_softmax as edge_softmax

class GATLayer(nn.Module):
    def __init__(self, 
                    input_vdim, 
                    input_edim, 
                    output_vdim, 
                    output_edim,
                    weight_dim=0,
                    num_heads=1, 
                    feat_drop=0, 
                    attn_drop=0, 
                    activation=None,
                    printflag=False, 
                    outputdir=""):
        super(GATLayer, self).__init__()
           
        self._input_vdim = input_vdim
        self._input_edim = input_edim
        self._output_vdim = output_vdim
        self._output_edim = output_edim
        self._weight_dim = weight_dim
        self._num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.activation = activation
        
        self.fc_ke = nn.Linear(input_vdim, output_edim * num_heads, bias=False) # vertex
        self.fc_qe = nn.Linear(input_edim, output_edim * num_heads, bias=False) # edge
        self.attn_ke = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_edim)))
        self.attn_qe = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_edim)))
        self.bias_e = nn.Parameter(torch.FloatTensor(size=(num_heads * output_edim,)))
        
        self.fc_kv = nn.Linear(output_edim, output_vdim * num_heads, bias=False) # edge
        self.fc_qv = nn.Linear(input_vdim, output_vdim * num_heads, bias=False) # vertex
        self.attn_kv = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_vdim)))
        self.attn_qv = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_vdim)))
        self.bias_v = nn.Parameter(torch.FloatTensor(size=(num_heads * output_edim,)))
        
        if self._weight_dim > 0:
            self.fc_we = nn.Linear(weight_dim, output_edim * num_heads, bias=False)
            self.attn_we = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_edim)))
            self.fc_wv = nn.Linear(weight_dim, output_vdim * num_heads, bias=False)
            self.attn_wv = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_vdim)))
        
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_ke.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_qe.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_kv.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_qv.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_ke, gain=gain)
        nn.init.xavier_normal_(self.attn_qe, gain=gain)
        nn.init.xavier_normal_(self.attn_kv, gain=gain)
        nn.init.xavier_normal_(self.attn_qv, gain=gain)
        if hasattr(self, 'fc_we'):
            nn.init.xavier_normal_(self.fc_we.weight, gain=gain)
        if hasattr(self, 'fc_wv'):
            nn.init.xavier_normal_(self.fc_wv.weight, gain=gain)
        if hasattr(self, 'attn_we'):
            nn.init.xavier_normal_(self.attn_we, gain=gain)
        if hasattr(self, 'attn_wv'):
            nn.init.xavier_normal_(self.attn_wv, gain=gain)
        nn.init.constant_(self.bias_e, 0.0)
        nn.init.constant_(self.bias_v, 0.0)
        
    def attention(self, edges):
        if self._weight_dim > 0:
            e = edges.src['el'] + edges.dst['er'] + edges.data['ew']
        else:
            e = edges.src['el'] + edges.dst['er']  
        
        return {'e': e}

    def forward(self, g1, g2, vfeat, efeat): # , hedgeorder, nodeorder
        with g1.local_scope():
            vfeat = self.feat_drop(vfeat)
            efeat = self.feat_drop(efeat[:g1['in'].num_dst_nodes()])
            
            vfeat_ke = self.fc_ke(vfeat).view(-1, self._num_heads, self._output_edim)
            efeat_qe = self.fc_qe(efeat).view(-1, self._num_heads, self._output_edim)
            el_e = (vfeat_ke * self.attn_ke).sum(dim=-1).unsqueeze(-1)
            er_e = (efeat_qe * self.attn_qe).sum(dim=-1).unsqueeze(-1)
            # self.curoutputname = self.outputname_e
            g1.srcnodes['node'].data['ft'] = vfeat_ke
            g1.srcnodes['node'].data['el'] = el_e
            g1.dstnodes['edge'].data['er'] = er_e
            if hasattr(self, 'attn_we') and hasattr(self, 'fc_we'):
                we = g1['in'].edata["weight"]
                feat_we = self.fc_we(we).view(-1, self._num_heads, self._output_edim)
                ew_e = (feat_we * self.attn_we).sum(dim=-1).unsqueeze(-1)
                g1['in'].edata['ew'] = ew_e
            g1.apply_edges(self.attention, etype='in')
            e = F.relu(g1['in'].edata.pop('e'))
            g1['in'].edata['a'] = self.attn_drop(edge_softmax(g1['in'], e))
            g1.update_all(fn.u_mul_e('ft', 'a', 'm'),
                        fn.sum('m', 'ft'), etype='in')
            efeat = g1['in'].dstdata['ft']
            efeat = efeat + self.bias_e.view(1, self._num_heads, self._output_edim)
            if self.activation:
                efeat = self.activation(efeat)
            efeat = efeat.mean(1)
            
        with g2.local_scope():
            efeat_kv = self.fc_kv(self.feat_drop(efeat)).view(-1, self._num_heads, self._output_vdim)
            vfeat_qv = self.fc_qv(vfeat[:g2['con'].num_dst_nodes()]).view(-1, self._num_heads, self._output_vdim)
            el_v = (efeat_kv * self.attn_kv).sum(dim=-1).unsqueeze(-1)
            er_v = (vfeat_qv * self.attn_qv).sum(dim=-1).unsqueeze(-1)
            # self.curoutputname = self.outputname_v
            g2['con'].srcdata.update({'ft': efeat_kv, 'el': el_v})
            g2['con'].dstdata.update({'er': er_v})
            if hasattr(self, 'attn_wv') and hasattr(self, 'fc_wv'):
                wv = g2['con'].edata["weight"]
                feat_wv = self.fc_wv(wv).view(-1, self._num_heads, self._output_vdim)
                ew_v = (feat_wv * self.attn_wv).sum(dim=-1).unsqueeze(-1)
                g2['con'].edata['ew'] = ew_v
            g2.apply_edges(self.attention, etype='con')
            e = F.relu(g2['con'].edata.pop('e'))
            g2['con'].edata['a'] = self.attn_drop(edge_softmax(g2['con'], e))
            g2.update_all(fn.u_mul_e('ft', 'a', 'm'),
                        fn.sum('m', 'ft'), etype='con')
            vfeat = g2['con'].dstdata['ft']
            vfeat = vfeat + self.bias_v.view(1, self._num_heads, self._output_vdim)
            if self.activation:
                vfeat = self.activation(vfeat)
            vfeat = vfeat.mean(1)
            
        return [vfeat, efeat]

class GAT(nn.Module):
    def __init__(self, 
                 input_vdim,
                 input_edim,
                 hidden_dim, 
                 vertex_dim,
                 edge_dim,
                 weight_dim=0,
                 num_layers=3, 
                 num_heads=4, 
                 feat_drop=0.6, 
                 attn_drop=0.6):
        super(GAT, self).__init__()
        
        self.activation = F.relu
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
                       
        if num_layers == 1:
             self.layers.append(GATLayer(input_vdim, input_edim, vertex_dim, edge_dim, 
                                         weight_dim=weight_dim, num_heads=1, feat_drop=0., attn_drop=0., activation=None))
        else:
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(GATLayer(input_vdim, input_edim, hidden_dim, hidden_dim,
                                                weight_dim=weight_dim, num_heads=num_heads, feat_drop=0., attn_drop=0., activation=self.activation))
                elif i == (num_layers - 1):
                    self.layers.append(GATLayer(hidden_dim, hidden_dim,vertex_dim, edge_dim,
                                                weight_dim=weight_dim, num_heads=1, feat_drop=feat_drop, attn_drop=attn_drop, activation=None))
                else:
                    self.layers.append(GATLayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim,
                                                weight_dim=weight_dim, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, activation=self.activation))
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
            
    def forward(self, blocks, vfeat, efeat):
        for l in range(self.num_layers):
            # print(vfeat.shape)
            # print(efeat.shape)
            vfeat, efeat = self.layers[l](blocks[2*l], blocks[2*l+1], vfeat, efeat)
        # print(vfeat.shape)
        # print(efeat.shape)
        return vfeat, efeat
    