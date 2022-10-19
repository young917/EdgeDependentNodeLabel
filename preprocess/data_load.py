import torch
import numpy as np
import math
from collections import defaultdict
import dgl
import utils
import random
import pickle
import os
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import train_test_split
from dgl import dataloading
from dgl import sampling, subgraph, distributed
from tqdm import tqdm, trange
import scipy.sparse as sp
import hashlib
from scipy.sparse.linalg import expm
from scipy.sparse import csr_matrix

def make_order(ls):
    a = np.array(ls)
    argsorted = np.argsort(a)
    orders = np.zeros(a.shape)
    a = sorted(a)
    
    # adjust
    previous = None
    order = 1
    for i, _a  in enumerate(a):
        if previous is None:
            previous = _a
            orders[argsorted[i]] = order
        elif previous != _a:
            order += 1
            orders[argsorted[i]] = order
            previous = _a
        else:
            orders[argsorted[i]] = order
    return orders.tolist()

class Hypergraph:
    def __init__(self, args, dataset_name):
        self.inputdir = args.inputdir
        self.dataname = dataset_name
        self.exist_hedgename = args.exist_hedgename
        self.valid_inputname = args.valid_inputname
        self.test_inputname = args.test_inputname
        self.use_gpu = args.use_gpu
        self.k = args.k
        
        self.hedge2node = []
        self.node2hedge = [] 
        self.hedge2nodepos = [] # hyperedge index -> node positions (after binning)
        self._hedge2nodepos = [] # hyperedge index -> node positions (before binning)
        self.node2hedgePE = []
        self.hedge2nodePE = []
        self.weight_flag = False
        self.hedge2nodeweight = []
        self.node2hedgeweight = []
        self.numhedges = 0
        self.numnodes = 0
        
        
        self.hedgeindex = {} # papaercode -> index
        self.hedgename = {} # index -> papercode
        self.e_feat = []

        self.node_reindexing = {} # nodeindex -> reindex
        self.node_orgindex = {} # reindex -> nodeindex
        self.v_feat = [] # (V, 1)
        
        self.load_graph(args)        
        print("Data is prepared")
        
    def load_graph(self, args):
        # construct connection  -------------------------------------------------------
        hset = []
        if args.k > 0:
            with open(self.inputdir + self.dataname + "/sampled_hset_" + str(args.k) + ".txt", "r") as f:
                for line in f.readlines():
                    line = line.rstrip()
                    hset.append(int(line))
        self.max_len = 0
        with open(self.inputdir + self.dataname + "/hypergraph.txt", "r") as f:
            for _hidx, line in enumerate(f.readlines()):
                if (args.k == 0) or ((args.k > 0) and (_hidx in hset)):
                    tmp = line.split("\t")
                    hidx = self.numhedges
                    self.numhedges += 1
                    if self.exist_hedgename:
                        papercode = tmp[0][1:-1] # without '
                        papercode = papercode.rstrip()
                        self.hedgeindex[papercode] = hidx
                        self.hedgename[hidx] = papercode
                        tmp = tmp[1:]
                    else:
                        self.hedgeindex[_hidx] = hidx
                        self.hedgename[hidx] = _hidx
                    self.hedgeindex[_hidx] = hidx
                    self.hedgename[hidx] = _hidx
                    self.hedge2node.append([])
                    self.hedge2nodepos.append([])
                    self._hedge2nodepos.append([])
                    self.hedge2nodePE.append([])
                    self.hedge2nodeweight.append([])
                    self.e_feat.append([])
                    if (self.max_len < len(tmp)):
                        self.max_len = len(tmp)
                    for node in tmp:
                        node = int(node.rstrip())
                        if node not in self.node_reindexing:
                            node_reindex = self.numnodes
                            self.numnodes += 1 
                            self.node_reindexing[node] = node_reindex
                            self.node_orgindex[node_reindex] = node 
                            self.node2hedge.append([])
                            self.node2hedgePE.append([])
                            self.node2hedgeweight.append([])
                            self.v_feat.append([])
                        nodeindex = self.node_reindexing[node]
                        self.hedge2node[hidx].append(nodeindex)
                        self.node2hedge[nodeindex].append(hidx)
                        self.hedge2nodePE[hidx].append([])
                        self.node2hedgePE[nodeindex].append([])
                    
        print("Max Size = ", self.max_len)
        print("Number of Hyperedges : " + str(self.numhedges))
        print("Number of Nodes : " + str(self.numnodes))
        # update by max degree
        for vhedges in self.node2hedge:
            if self.max_len < len(vhedges):
                self.max_len = len(vhedges)
        self.v_feat = torch.tensor(self.v_feat).type('torch.FloatTensor')
        for h in range(len(self.e_feat)):
            self.e_feat[h] = [0 for _ in range(args.dim_edge)]
        self.e_feat = torch.tensor(self.e_feat).type('torch.FloatTensor')
        
        # Split Data ------------------------------------------------------------------------
        self.test_index = []
        self.valid_index = []
        self.validsize = 0
        self.testsize = 0
        self.trainsize = 0
        self.hedge2type = torch.zeros(self.numhedges)
        
        assert os.path.isfile(self.inputdir + self.dataname + "/" + self.valid_inputname + "_" + str(self.k) + ".txt")
        with open(self.inputdir + self.dataname + "/" + self.valid_inputname + "_" + str(self.k) + ".txt", "r") as f:
            for line in f.readlines():
                name = line.rstrip()
                if self.exist_hedgename is False:
                    name = int(name)
                index = self.hedgeindex[name]
                self.valid_index.append(index)
            self.hedge2type[self.valid_index] = 1
            self.validsize = len(self.valid_index)
        if os.path.isfile(self.inputdir + self.dataname + "/" + self.test_inputname + "_" + str(self.k) + ".txt"):
            with open(self.inputdir + self.dataname + "/" + self.test_inputname + "_" + str(self.k) + ".txt", "r") as f:
                for line in f.readlines():
                    name = line.rstrip()
                    if self.exist_hedgename is False:
                        name = int(name)
                    index = self.hedgeindex[name]
                    self.test_index.append(index)
                assert len(self.test_index) > 0
                self.hedge2type[self.test_index] = 2
                self.testsize = len(self.test_index)
        self.trainsize = self.numhedges - (self.validsize + self.testsize)
        
        # extract target ---------------------------------------------------------
        print("Extract labels")
        with open(self.inputdir + self.dataname + "/hypergraph_pos.txt", "r") as f:
            for _hidx, line in enumerate(f.readlines()):
                tmp = line.split("\t")
                if self.exist_hedgename:
                    papercode = tmp[0][1:-1] # without ''
                    if (papercode not in self.hedgeindex):
                        continue
                    hidx = self.hedgeindex[papercode]
                    tmp = tmp[1:]
                else:
                    if (_hidx not in self.hedgeindex):
                        continue
                    hidx = self.hedgeindex[_hidx]
                if args.binning > 0:
                    positions = [float(i) for i in tmp]
                    for nodepos in positions:
                        self._hedge2nodepos[hidx].append(nodepos)
                else:
                    positions = [int(i) for i in tmp]
                for nodepos in positions:
                    self.hedge2nodepos[hidx].append(nodepos)
        # labeled by binning
        if args.binning > 0:
            weights = sorted([w for h in self.get_data(type=0) for w in self._hedge2nodepos[h]])
            total_num = len(weights)
            cum = 0
            self.binindex = []
            for w in weights:
                cum += 1
                if (cum / total_num) >=  ((1.0 / args.binning) * (len(self.binindex) + 1)):
                    self.binindex.append(w)
            print("BinIndex", self.binindex)
            with open(self.inputdir + self.dataname + "/binindex.txt", "w") as f:
                for binvalue in self.binindex:
                    f.write(str(binvalue) + "\n")
            # float -> int
            for h in range(self.numhedges):
                for i, w in enumerate(self._hedge2nodepos[h]):
                    for bi, bv in enumerate(self.binindex):
                        if w <= bv:
                            self.hedge2nodepos[h][i] = bi
                            break
                        elif bi == (args.output_dim - 1) and w > bv:
                            self.hedge2nodepos[h][i] = bi
                            break
            # check
            for h in range(self.numhedges):
                for w in self.hedge2nodepos[h]:
                    assert w in range(args.binning), str(w)
        
        # extract PE ----------------------------------------------------------------------------------------------------
        # hedge2nodePE
        if len(args.vorder_input) > 0: # centrality -> PE ------------------------------------------------------------------
            self.order_dim = len(args.vorder_input)
            for inputpath in args.vorder_input:
                vfeat = {} # node -> vfeat
                with open(self.inputdir + self.dataname + "/" + inputpath + "_" + str(args.k) + ".txt", "r") as f:
                    columns = f.readline()
                    columns = columns[:-1].split("\t")
                    for line in f.readlines():
                        line = line.rstrip()
                        tmp = line.split("\t")
                        nodeindex = int(tmp[0])
                        if nodeindex not in self.node_reindexing:
                            # not include in incidence matrix
                            continue
                        node_reindex = self.node_reindexing[nodeindex]
                        for i, col in enumerate(columns):
                            vfeat[node_reindex] = float(tmp[i])
                if args.whole_order: # in entire nodeset
                    feats = []
                    for vidx in range(self.numnodes):
                        feats.append(vfeat[vidx])
                    orders = make_order(feats)
                    for hidx, hedge in enumerate(self.hedge2node):
                        for vorder, v in enumerate(hedge):
                            self.hedge2nodePE[hidx][vorder].append((orders[v]) / self.numnodes)
                else: # in each hyperedge
                    for hidx, hedge in enumerate(self.hedge2node):
                        feats = []
                        for v in hedge:
                            feats.append(vfeat[v])
                        orders = make_order(feats)
                        for vorder, v in enumerate(hedge):
                            self.hedge2nodePE[hidx][vorder].append((orders[vorder]) / len(feats))
            # check            
            assert len(self.hedge2nodePE) == self.numhedges
            for hidx in range(self.numhedges):
                assert len(self.hedge2nodePE[hidx]) == len(self.hedge2node[hidx])
                for vorder in self.hedge2nodePE[hidx]:
                    assert len(vorder) == len(args.vorder_input)
            # node2hedgePE
            for vidx, node in enumerate(self.node2hedge):
                orders = []
                for hidx in node:
                    for vorder,_v in enumerate(self.hedge2node[hidx]):
                        if _v == vidx:
                            orders.append(self.hedge2nodePE[hidx][vorder])
                            break
                self.node2hedgePE[vidx] = orders
            # check
            assert len(self.node2hedgePE) == self.numnodes
            for vidx in range(self.numnodes):
                assert len(self.node2hedgePE[vidx]) == len(self.node2hedge[vidx])
                for horder in self.node2hedgePE[vidx]:
                    assert len(horder) == len(args.vorder_input)
            self.weight_flag = True
        elif len(args.pe) > 0: # ---------------------------------------------------------------------------------------------
            # Use other positional encoding!
            rows, cols = [], [] # construct adjacency matrix
            for v in range(self.numnodes):
                hedges = self.node2hedge[v]
                check = np.zeros(self.numnodes)
                for h in hedges:
                    neighbors = self.hedge2node[h]
                    for nv in neighbors:
                        if v < nv and check[nv] == 0:
                            check[nv] = 1
                            rows.append(v)
                            cols.append(nv)
                            rows.append(nv)
                            cols.append(v)
            A = sp.coo_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols))), shape=(self.numnodes, self.numnodes))
            _deg = A.sum(axis=1).squeeze(1)
            deg = list(_deg.flat)
            deg = np.array(deg)
            print("Adj is prepared")
            if args.pe in ["DK", "PRWK"]:
                # sorting hedge2node, hedge2nodepos
                for hidx in range(self.numhedges):
                    sorted_idx = np.argsort(np.array(self.hedge2node[hidx]))
                    self.hedge2node[hidx] = np.array(self.hedge2node[hidx])[sorted_idx].tolist()
                    self.hedge2nodepos[hidx] = np.array(self.hedge2nodepos[hidx])[sorted_idx].tolist()
                if args.pe == "DK":
                    L = sp.diags(deg, dtype=float) - A # No Normalize
                    beta = 1.0
                    L = -beta * L
                    v2v = L 
                    for hidx in trange(self.numhedges, desc="making KD per hedge"):
                        hedge = self.hedge2node[hidx]
                        _v2v_e = []
                        for vidx in range(len(hedge)):
                            vi = hedge[vidx]
                            _row = v2v.getrow(vi).toarray()[0]
                            efeat = []
                            for nvidx in range(len(hedge)):
                                nv = hedge[nvidx]
                                if (_row[nv] < 0 and vidx != nvidx):
                                    assert vi == nv
                                    efeat.append(1.0)
                                else:
                                    efeat.append(_row[nv])
                            _v2v_e.append(efeat)
                        _v2v_e = np.array(_v2v_e)
                        v2v_e = expm(_v2v_e)
                        for vorder in range(len(hedge)):
                            self.hedge2nodePE[hidx][vorder] = v2v_e[vidx].tolist()
                            for _pe in self.hedge2nodePE[hidx][vorder]:
                                if _pe < 0:
                                    print(_v2v_e[vidx])
                                    print(v2v_e[vidx])
                                    print(self.hedge2nodePE[hidx][vorder])
                                assert _pe >= 0
                    
                elif args.pe == "PRWK":
                    L = sp.diags(deg, dtype=float) - A # No Normalize
                    print("L is prepared")
                    gamma, p = 0.5, 2
                    r = sp.eye(self.numnodes) - gamma * L
                    print("R is prepared")
                    v2v = r.power(p) # |V|x|V|
                
                    for hidx in range(self.numhedges):
                        hedge = self.hedge2node[hidx]
                        for vidx in range(len(hedge)):
                            vi = hedge[vidx]
                            efeat = []
                            for nvidx in range(len(hedge)):
                                nv = hedge[nvidx]
                                _row = v2v.getrow(vi).toarray()[0]
                                efeat.append(_row[nv])
                            self.hedge2nodePE[hidx][vidx] = efeat
                            for _pe in self.hedge2nodePE[hidx][vidx]:
                                assert _pe >= 0
                self.weight_flag = True
        
        # For HGNN ----------------------------------------------------------------------------------
        if args.embedder == "hgnn" or args.embedder == "hcha":
            nodedeg = []
            hedgedeg = []
            for hedges in self.node2hedge:
                nodedeg.append(len(hedges))
            for nodes in self.hedge2node:
                hedgedeg.append(len(nodes))
            self.DV2 = torch.pow(torch.FloatTensor(nodedeg), -0.5)
            self.invDE = torch.pow(torch.FloatTensor(hedgedeg),-1)
        # applying alpha and beta in HNHN ---------------------------------------------------------
        print("weight")
        e_weight = []
        v_weight = []
        for neighbor_hedges in self.node2hedge:
            v_weight.append(len(neighbor_hedges))
        for hedge in self.hedge2node:
            e_weight.append(len(hedge))
        use_exp_wt = args.use_exp_wt
        e_reg_weight = torch.zeros(self.numhedges)
        v_reg_weight = torch.zeros(self.numnodes)
        for hidx in range(self.numhedges):
            e_wt = e_weight[hidx]
            e_reg_wt = torch.exp(args.alpha_e*e_wt) if use_exp_wt else e_wt**args.alpha_e 
            e_reg_weight[hidx] = e_reg_wt
        for vidx in range(self.numnodes):
            v_wt = v_weight[vidx]
            v_reg_wt = torch.exp(args.alpha_v*v_wt) if use_exp_wt else v_wt**args.alpha_v
            v_reg_weight[vidx] = v_reg_wt
        v_reg_sum = torch.zeros(self.numnodes) # <- e_reg_weight2v_sum
        e_reg_sum = torch.zeros(self.numhedges) # <- v_reg_weight2e_sum
        for hidx, hedges in enumerate(self.hedge2node):
            for vidx in hedges:
                v_reg_sum[vidx] += e_reg_wt
                e_reg_sum[hidx] += v_reg_wt  
        e_reg_sum[e_reg_sum==0] = 1
        v_reg_sum[v_reg_sum==0] = 1
        self.e_reg_weight = torch.Tensor(e_reg_weight).unsqueeze(-1)
        self.v_reg_sum = torch.Tensor(v_reg_sum).unsqueeze(-1)
        self.v_reg_weight = torch.Tensor(v_reg_weight).unsqueeze(-1)
        self.e_reg_sum = torch.Tensor(e_reg_sum).unsqueeze(-1)
        # check
        for hidx, hedges in enumerate(self.hedge2node):
            e_reg_sum = self.e_reg_sum[hidx]
            v_reg_sum = 0
            for vidx in hedges:
                v_reg_sum += self.v_reg_weight[vidx]
            assert abs(e_reg_sum - v_reg_sum) < 1e-4
        # UniGCNII ----------------------------------------------------------------------------------
        if args.embedder == "unigcnii":
            degV = []
            for vidx, hedges in enumerate(self.node2hedge):
                degV.append(len(hedges))
            degE = []
            for eidx, nodes in enumerate(self.hedge2node):
                avgdeg = 0
                for v in nodes:
                    avgdeg += (degV[v] / len(nodes))
                degE.append(avgdeg)
            self.degV = torch.Tensor(degV).pow(-0.5).unsqueeze(-1)
            self.degE = torch.Tensor(degE).pow(-0.5).unsqueeze(-1)

    def get_data(self, type=0):
        hedgelist = ((self.hedge2type == type).nonzero(as_tuple=True)[0])
        if self.use_gpu is False:
            hedgelist = hedgelist.tolist()
        return hedgelist
    
# Generate DGL Graph ==============================================================================================
def gen_DGLGraph(args, hedge2node, hedge2nodepos, node2hedge, device):
    data_dict = defaultdict(list)
    in_edge_label = []
    con_edge_label = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            in_edge_label.append(hedge2nodepos[hidx][vorder]) 
            con_edge_label.append(hedge2nodepos[hidx][vorder]) 

    in_edge_label = torch.Tensor(in_edge_label)
    con_edge_label = torch.Tensor(con_edge_label)

    g = dgl.heterograph(data_dict)
    g['in'].edata['label'] = in_edge_label
    g['con'].edata['label'] = con_edge_label
    return g

def gen_weighted_DGLGraph(args, hedge2node, hedge2nodePE, hedge2nodepos, node2hedge, node2hedgeorder, device):
    edgefeat_dim = 0
    for efeat_list in hedge2nodePE:
        efeat_dim = len(efeat_list[0])
        edgefeat_dim = max(edgefeat_dim, efeat_dim)
    print("Edge Feat Dim ", edgefeat_dim)
    
    data_dict = defaultdict(list)
    in_edge_weights = []
    in_edge_label = []
    con_edge_weights = []
    con_edge_label = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            # connection
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            # edge feat
            efeat = hedge2nodePE[hidx][vorder]
            efeat += np.zeros(edgefeat_dim - len(efeat)).tolist()
            in_edge_weights.append(efeat)
            con_edge_weights.append(efeat)
            # label
            in_edge_label.append(hedge2nodepos[hidx][vorder])
            con_edge_label.append(hedge2nodepos[hidx][vorder])

    in_edge_weights = torch.Tensor(in_edge_weights)
    con_edge_weights = torch.Tensor(con_edge_weights)
    in_edge_label = torch.Tensor(in_edge_label)
    con_edge_label = torch.Tensor(con_edge_label)

    g = dgl.heterograph(data_dict)
    g['in'].edata['weight'] = in_edge_weights
    g['con'].edata['weight'] = con_edge_weights
    g['in'].edata['label'] = in_edge_label
    g['con'].edata['label'] = con_edge_label
    
    return g
