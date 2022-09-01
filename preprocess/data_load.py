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

def make_ranking(ls):
    a = np.array(ls)
    argsorted = np.argsort(a)
    ranking = np.zeros(a.shape)
    # print(ranking)
    a = sorted(a)
    # print(a)

    # adjust
    previous = None
    rank = 1
    for i, _a  in enumerate(a):
        if previous is None:
            previous = _a
            ranking[argsorted[i]] = rank
        elif previous != _a:
            rank += 1
            ranking[argsorted[i]] = rank
            previous = _a
        else:
            ranking[argsorted[i]] = rank
    return ranking.tolist()

class Hypergraph:
    def __init__(self, args, dataset_name):
        self.inputdir = args.inputdir
        self.dataname = dataset_name
        self.exist_hedgename = args.exist_hedgename
        self.valid_inputname = args.valid_inputname
        self.test_inputname = args.test_inputname
        self.use_gpu = args.use_gpu
        self.k = args.k
        self.n_cls = 3

        self.hedge2node = [] 
        self.node2hedge = [] 
        self.hedge2nodepos = [] # hyperedge index -> node positions
        self.node2hedgerank = []
        self.hedge2noderank = []
        self.hedge2nodeweight = []
        self.node2hedgeweight = []
        self.numhedges = 0
        self.numnodes = 0
        
        self.hedgeindex = {} # papaercode -> index
        self.hedgename = {} # index -> papercode
        self.e_feat = [] # (To be revised) all ones... (E, 1)
        self.e_feat_input = args.efeat_input
        self.e_feat_columns = args.efeat_usecols

        self.node_reindexing = {} # nodeindex -> reindex
        self.node_orgindex = {} # reindex -> nodeindex
        self.v_feat = [] # (V, 1)
        self.v_feat_input = args.vfeat_input
        self.v_feat_columns = args.vfeat_usecols

        self.load_graph(args)
        self.hedge2type = torch.zeros(self.numhedges) #[0 for _ in range(self.numhedges)] # 0 - train, 1 - val, 2 - test
        
        # self.v_weight (V, 1)
        # self.e_weight (E, 1)
        # self.e_reg_weight (E, 1)
        # self.v_reg_sum (V, 1)
        # self.v_reg_weight (V, 1)
        # self.e_reg_sum (E, 1)
        
        print("Data is prepared")
        
    def load_graph(self, args):
        hset = []
        if args.k > 0:
            with open(self.inputdir + self.dataname + "/sampled_hset_" + str(args.k) + ".txt", "r") as f:
                for line in f.readlines():
                    line = line.rstrip()
                    hset.append(int(line))
                    
        self.max_len = 0
        # construct connection  -------------------------------------------------------
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
                    self.hedge2node.append([])
                    self.hedge2nodepos.append([])
                    self.hedge2noderank.append([])
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
                            self.node2hedgerank.append([])
                            self.node2hedgeweight.append([])
                            self.v_feat.append([])
                        nodeindex = self.node_reindexing[node]
                        self.hedge2node[hidx].append(nodeindex)
                        self.node2hedge[nodeindex].append(hidx)
                        self.hedge2noderank[hidx].append([])
                        self.node2hedgerank[nodeindex].append([])
                    
        print("Max Size = ", self.max_len)
        print("Number of Hyperedges : " + str(self.numhedges))
        print("Number of Nodes : " + str(self.numnodes))
        # update by max degree
        for vhedges in self.node2hedge:
            if self.max_len < len(vhedges):
                self.max_len = len(vhedges)
        
        # For HGNN
        if args.embedder == "hgnn" or args.embedder == "hcha":
#             try:
#                 incidence = torch.zeros(self.numhedges, self.numnodes)
#                 for e, hedge in enumerate(self.hedge2node):
#                     for v in hedge:
#                         incidence[e, v] = 1
#                 H = torch.tensor(incidence.T)
#                 self.DV2 = torch.pow(torch.sum(H, axis=1), -0.5)
#                 self.invDE = torch.pow(torch.sum(H, axis=0), -1)
#             except:
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
        if len(args.use_vweight_input) > 0:
            tmp = {}
            with open(self.inputdir + args.dataset_name + "/" + args.use_vweight_input + "_nodecentrality_" + str(args.k) + ".txt", "r") as f:
                f.readline()
                for line in f.readlines():
                    line = line.rstrip()
                    nodeindex, weight = line.split("\t")
                    nodeindex = int(nodeindex)
                    weight = float(weight)
                    if nodeindex in self.node_reindexing:
                        node_reindex = self.node_reindexing[nodeindex]
                        tmp[node_reindex] = weight
            for v in range(self.numnodes):
                v_weight.append(tmp[v])
        else:
            for neighbor_hedges in self.node2hedge:
                v_weight.append(len(neighbor_hedges))
        
        if len(args.use_eweight_input) > 0:
            tmp = {}
            with open("dataset/" + args.dataset_name + "/" + args.use_eweight_input + "_hfeatcentrality_" + str(args.k) + ".txt", "r") as f:
                for line in f.readlines():
                    line = line.rstrip()
                    hedgename, weight = line.split("\t")
                    weight = float(weight)
                    if hedgename in self.hedgeindex:
                        hidx = self.hedgeindex[hedgename]
                        tmp[hidx] = weight
                    elif int(hedgename) in self.hedgeindex:
                        hidx = self.hedgeindex[int(hedgename)]
                        tmp[hidx] = weight
            for h in range(self.numhedges):
                e_weight.append(tmp[h])
        else:
            for hedge in self.hedge2node:
                e_weight.append(len(hedge))
        
        use_exp_wt = args.use_exp_wt
        e_reg_weight = torch.zeros(self.numhedges)
        v_reg_weight = torch.zeros(self.numnodes)
        e_reg_weight2v_sum = defaultdict(list)
        v_reg_weight2e_sum = defaultdict(list)
        for hidx in range(self.numhedges):
            e_wt = e_weight[hidx]
            e_reg_wt = torch.exp(args.alpha_e*e_wt) if use_exp_wt else e_wt**args.alpha_e 
            e_reg_weight[hidx] = e_reg_wt
        for vidx in range(self.numnodes):
            v_wt = v_weight[vidx]
            v_reg_wt = torch.exp(args.alpha_v*v_wt) if use_exp_wt else v_wt**args.alpha_v
            v_reg_weight[vidx] = v_reg_wt
        for hidx, hedges in enumerate(self.hedge2node):
            for vidx in hedges:
                # Isn't it right?
                e_reg_weight2v_sum[vidx].append(e_reg_wt)
                v_reg_weight2e_sum[hidx].append(v_reg_wt)   
        v_reg_sum = torch.zeros(self.numnodes) # <- e_reg_weight2v_sum
        e_reg_sum = torch.zeros(self.numhedges) # <- v_reg_weight2e_sum
        for vidx, wt_l in e_reg_weight2v_sum.items():
            v_reg_sum[vidx] = sum(wt_l)
        for hidx, wt_l in v_reg_weight2e_sum.items():
            e_reg_sum[hidx] = sum(wt_l)
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
                positions = [int(i) for i in tmp]
                for nodepos in positions:
                    self.hedge2nodepos[hidx].append(nodepos)
                assert len(self.hedge2nodepos[hidx]) == len(self.hedge2node[hidx])
        
        # extract features ---------------------------------------------------------
        print("Save v_feat and e_feat: ", args.vfeat_input, args.efeat_input)
        for inputpath, binning_flag in zip(args.vfeat_input, args.binnings_v):
            if inputpath == 'identity':
                for v in range(self.numnodes):
                    f = [0 for _ in range(self.numnodes)]
                    f[v] = 1
                    self.v_feat[v] += f
            else:
                if inputpath == "nodeinfo": # given feature
                    with open(self.inputdir + self.dataname + "/" + inputpath + "_" + str(args.k) + ".txt", "r", encoding="utf-8") as f:
                        if binning_flag:
                            line = f.readline().rstrip()
                            _, length = line.split("\t")
                            length = int(length)
                            print("Length:", length)
                        columns = f.readline()
                        columns = columns[:-1].split("\t")
                        print(columns, self.v_feat_columns)
                        for line in f.readlines():
                            line = line.rstrip()
                            tmp = line.split("\t")
                            nodeindex = int(tmp[0])
                            if nodeindex not in self.node_reindexing:
                                # not include in incidence matrix
                                continue
                            node_reindex = self.node_reindexing[nodeindex]
                            for i, col in enumerate(columns):
                                if col in self.v_feat_columns:
                                    if tmp[i] == 'NULL':
                                        tmp[i] = 0
                                    if binning_flag:
                                        l = [0 for idx in range(length)]
                                        l[int(tmp[i])] = 1
                                        self.v_feat[node_reindex] += l
                                    else:
                                        self.v_feat[node_reindex].append(float(tmp[i]))
                else: # feature extracted in advance
                    with open(self.inputdir + self.dataname + "/" + inputpath + "_" + str(args.k) + ".txt", "r") as f:
                        if binning_flag:
                            line = f.readline().rstrip()
                            _, length = line.split("\t")
                            length = int(length)
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
                                if col in self.v_feat_columns:
                                    if tmp[i] == 'NULL':
                                        tmp[i] = 0
                                    if binning_flag:
                                        l = [0 for idx in range(length)]
                                        l[int(tmp[i])] = 1
                                        self.v_feat[node_reindex] += l
                                    else:
                                        self.v_feat[node_reindex].append(float(tmp[i]))
#                 if binning_flag is False:
#                     self.v_feat = stats.zscore(self.v_feat, axis=0, ddof=1) # normalize
        
        self.v_feat = torch.tensor(self.v_feat).type('torch.FloatTensor')
        

        for inputpath, binning_flag in zip(args.efeat_input, args.binnings_e):
            if inputpath == "hyperedgeinfo": # given hyperedge feature
                with open(self.inputdir + self.dataname + "/" + inputpath + "_" + str(args.k) + ".txt", "r", encoding="utf-8") as f:
                    if binning_flag:
                        line = f.readline().rstrip()
                        _, length = line.split("\t")
                        length = int(length)
                        print("Length:", length)
                    columns = f.readline().rstrip()
                    columns = columns.split("\t")
                    for line in f.readlines():
                        line = line.rstrip()
                        tmp = line.split("\t")
                        papercode = tmp[0]
                        if papercode not in self.hedgeindex:
                            # not include in incidence matrix
                            continue
                        hidx = self.hedgeindex[papercode]
                        for i, col in enumerate(columns):
                            if col in self.e_feat_columns:
                                if tmp[i] == 'NULL':
                                    tmp[i] = 0
                                if binning_flag:
                                    l = [0 for idx in range(length)]
                                    l[int(tmp[i])] = 1
                                    self.e_feat[hidx] += l
                                else:
                                    self.e_feat[hidx].append(float(tmp[i]))
            else:
                with open(self.inputdir + self.dataname + "/" + inputpath + "_" + str(args.k) + ".txt", "r") as f:
                    if binning_flag:
                        line = f.readline().rstrip()
                        _, length = line.split("\t")
                        length = int(length)
                    for line in f.readlines():
                        line = line.rstrip()
                        hedgename, val = line.split("\t")
                        if self.exist_hedgename is False:
                            hedgename = int(hedgename)
                        if hedgename not in self.hedgeindex:
                            # not include in incidence matrix
                            continue
                        hidx = self.hedgeindex[hedgename]
                        if binning_flag:
                            l = [0 for idx in range(length)]
                            l[int(val)] = 1
                            self.e_feat[hidx] += l
                        else:
                            self.e_feat[hidx].append(float(val))

        if len(self.e_feat[0]) == 0:
            for h in range(len(self.e_feat)):
                self.e_feat[h] = [0 for _ in range(args.dim_edge)]
        self.e_feat = torch.tensor(self.e_feat).type('torch.FloatTensor')
        print("Feature Dimension:")
        print(self.v_feat.shape)
        print(self.e_feat.shape)
        print(".")
        
        # extract rank ----------------------------------------------------------------------------------------------------
        # hedge2noderank
        self.rank_dim = len(args.vrank_input)
        for inputpath in args.vrank_input:
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
            if args.whole_ranking:
                feats = []
                for vidx in range(self.numnodes):
                    feats.append(vfeat[vidx])
                rankings = make_ranking(feats)
                for hidx, hedge in enumerate(self.hedge2node):
                    for vorder, v in enumerate(hedge):
                        self.hedge2noderank[hidx][vorder].append((rankings[v]) / self.numnodes)
            else: # in each hyperedge
                for hidx, hedge in enumerate(self.hedge2node):
                    feats = []
                    for v in hedge:
                        feats.append(vfeat[v])
                    rankings = make_ranking(feats)
                    for vorder, v in enumerate(hedge):
                        self.hedge2noderank[hidx][vorder].append((rankings[vorder]) / len(feats))
        # check            
        assert len(self.hedge2noderank) == self.numhedges
        for hidx in range(self.numhedges):
            assert len(self.hedge2noderank[hidx]) == len(self.hedge2node[hidx])
            for vrank in self.hedge2noderank[hidx]:
                assert len(vrank) == len(args.vrank_input)
        # node2hedgerank
        if len(args.erank_input) == 0:
            for vidx, node in enumerate(self.node2hedge):
                ranks = []
                for hidx in node:
                    for vorder,_v in enumerate(self.hedge2node[hidx]):
                        if _v == vidx:
                            ranks.append(self.hedge2noderank[hidx][vorder])
                            break
                self.node2hedgerank[vidx] = ranks
        else:
            for inputpath in args.erank_input:
                efeat = {}
                with open(self.inputdir + self.dataname + "/" + inputpath + "_" + str(args.k) + ".txt", "r") as f:
                    for line in f.readlines():
                        line = line.rstrip()
                        hedgename, val = line.split("\t")
                        if self.exist_hedgename is False:
                            hedgename = int(hedgename)
                        if hedgename not in self.hedgeindex:
                            # not include in incidence matrix
                            continue
                        hedgeindex = self.hedgeindex[hedgename]
                        efeat[hedgeindex] = float(val)
                if args.whole_ranking:
                    feats = []
                    for hidx in range(self.numhedges):
                        feats.append(efeat[hidx])
                    rankings = make_ranking(feats)
                    for vidx, node in enumerate(self.node2hedge):
                        for horder, h in enumerate(node):
                            self.node2hedgerank[vidx][horder].append((rankings[h]) / self.numhedges)
                else:
                    for vidx, node in enumerate(self.node2hedge):
                        feats = []
                        for h in node:
                            feats.append(efeat[h])
                        rankings = make_ranking(feats)
                        for horder, h in enumerate(node):
                            self.node2hedgerank[vidx][horder].append((rankings[horder]) / len(feats))
            
        # check
        assert len(self.node2hedgerank) == self.numnodes
        for vidx in range(self.numnodes):
            assert len(self.node2hedgerank[vidx]) == len(self.node2hedge[vidx])
            for hrank in self.node2hedgerank[vidx]:
                assert len(hrank) == len(args.vrank_input)
        
        # make sampling weight ----------------------------------------------------------------------------------------------------
        if args.use_sample_wt:
            for hidx, hedge in enumerate(self.hedge2node):
                nodeweights = [np.mean(rank) for rank in self.hedge2noderank[hidx]]
                if np.std(nodeweights) == 0:
                    nodeweights = [1 for _ in range(len(nodeweights))]
                else:
                    nodeweights = stats.zscore(nodeweights)
                    for w in nodeweights:
                        assert math.isnan(w) is False
                    nodeweights = [(abs(w) + 1) for w in nodeweights]
                self.hedge2nodeweight[hidx] = nodeweights
            for vidx, node in enumerate(self.node2hedge):
                hedgeweights = [np.mean(rank) for rank in self.node2hedgerank[vidx]]
                if np.std(hedgeweights) == 0:
                    hedgeweights = [1 for _ in range(len(hedgeweights))]
                else:
                    hedgeweights = stats.zscore(hedgeweights)
                    for w in hedgeweights:
                        assert math.isnan(w) is False
                    hedgeweights = [(abs(w) + 1) for w in hedgeweights]
                self.node2hedgeweight[vidx] = hedgeweights
    
    def split_data(self, val_ratio, test_ratio, seed=2022):
        test_index = []
        valid_index = []
#         if os.path.isfile("dataset/" + self.dataname + "/" + self.test_inputname + "_" + str(self.k) + ".txt") is False:
#         indexes = list(range(self.numhedges))
#         train_index, test_index = train_test_split(indexes, test_size = 0.2, random_state = 21)
#         test_hedgename = [self.hedgename[i] for i in test_index]
#         with open("dataset/" + self.dataname + "/test_hindex_" + str(self.k) + ".txt", "w") as f:
#             for name in test_hedgename:
#                 f.write(str(name) + "\n")
        if val_ratio > 0:
            with open(self.inputdir + self.dataname + "/" + self.valid_inputname + "_" + str(self.k) + ".txt", "r") as f:
                for line in f.readlines():
                    name = line.rstrip()
                    if self.exist_hedgename is False:
                        name = int(name)
                    index = self.hedgeindex[name]
                    valid_index.append(index)
        if test_ratio > 0:
            with open(self.inputdir + self.dataname + "/" + self.test_inputname + "_" + str(self.k) + ".txt", "r") as f:
                for line in f.readlines():
                    name = line.rstrip()
                    if self.exist_hedgename is False:
                        name = int(name)
                    index = self.hedgeindex[name]
                    test_index.append(index)
        # train_index = sorted(list(set(range(self.numhedges)) - set(test_index)))
            assert len(test_index) > 0
        
        self.hedge2type[test_index] = 2
        self.hedge2type[valid_index] = 1
        #for hidx in trange(self.numhedges):
            #if hidx in test_index:
            #    self.hedge2type[hidx] = 2
            #elif hidx in valid_index:
            #    self.hedge2type[hidx] = 1
            #else:
            #    self.hedge2type[hidx] = 0
        self.validsize = len(valid_index)
        self.testsize = len(test_index)
    
    # Revised
    def get_data(self, type=0):
        hedgelist = ((self.hedge2type == type).nonzero(as_tuple=True)[0])
        if self.use_gpu is False:
            hedgelist = hedgelist.tolist()
        #hedgelist = []
        #for h in range(self.numhedges):
        #    if self.hedge2type[h] == type:
        #        hedgelist.append(h)
        return hedgelist

    def make_pe(self, pe_type):
        # adjacency matrix
        rows, cols = [], []
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
        if pe_type == "KD":
            L = sp.diags(deg, dtype=float) - A # No Normalize
            beta = 1.0
            L = beta * L
            self.positional_encoding = L
            # self.positional_encoding = torch.Tensor(expm(beta * L)).type('torch.FloatTensor')
        elif pe_type == "KPRW":
            L = sp.diags(deg, dtype=float) - A # No Normalize
            print("L is prepared")
            gamma, p = 0.5, 2
            r = sp.eye(self.numnodes) - gamma * L
            print("R is prepared")
            self.positional_encoding = r.power(p)
            # self.positional_encoding = torch.Tensor().type('torch.FloatTensor') 
        elif pe_type == "DESPD":
            self.positional_encoding = A.power(2)
            # self.positonal_encoding = torch.Tensor(A.power(2)).type('torch.FloatTensor')
        elif pe_type == "DERW":
            _deg = A.sum(axis=1).squeeze(1)
            deg = list(_deg.flat)
            deg = np.array(deg)
            deg[deg > 0.0] = deg[deg > 0.0] ** (-1.0)
            N = sp.diags(deg, dtype=float)
            TP = N * A
            print("TP is prepared")
            self.positional_encoding = TP.power(3)
            # self.positional_encoding = torch.Tensor(TP.power(3)).type('torch.FloatTensor') 
        else:
            self.positional_encodng = None
            
    def get_pe(self, node_indices, pe_type):
        pe = torch.zeros((len(node_indices), self.numnodes),dtype=torch.float)
        for i, v in enumerate(node_indices):
            row = self.positional_encoding.getrow(v)
            _, col = row.nonzero()
            data = row.data
            for j, d in zip(col, data):
                pe[i][j] = d
        if pe_type == "KD":
            pe = torch.exp(pe)
        return pe
    
# Generate DGL Graph ==============================================================================================
def gen_DGLGraph(args, hedge2node, hedge2nodepos, node2hedge, device):
    data_dict = defaultdict(list)
    in_edge_label = []
    con_edge_label = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            in_edge_label.append(hedge2nodepos[hidx][vorder]) # hedge2nodepos[hidx][vorder]
            con_edge_label.append(hedge2nodepos[hidx][vorder]) # hedge2nodepos[hidx][vorder]

    in_edge_label = torch.Tensor(in_edge_label)
    con_edge_label = torch.Tensor(con_edge_label)

    g = dgl.heterograph(data_dict)
    g['in'].edata['label'] = in_edge_label
    g['con'].edata['label'] = con_edge_label
    return g

def gen_weighted_DGLGraph(args, hedge2node, hedge2noderank, hedge2nodepos, node2hedge, node2hedgerank, device):
    data_dict = defaultdict(list)
    in_edge_weights = []
    in_edge_label = []
    con_edge_weights = []
    con_edge_label = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            in_edge_weights.append(hedge2noderank[hidx][vorder])
            con_edge_weights.append(hedge2noderank[hidx][vorder])
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
'''    
def gen_DGLGraph(args, ground, device):
    he = []
    hv = []
    for i, edge in enumerate(ground):
        for v in edge :
            he.append(i)
            hv.append(v)
    data_dict = {
        ('node', 'in', 'edge'): (hv, he),        
        ('edge', 'con', 'node'): (he, hv)
    }
    g = dgl.heterograph(data_dict)
    return g

def gen_weighted_DGLGraph(args, hedge2node, hedge2noderank, node2hedge, node2hedgerank, device):
    data_dict = defaultdict(list)
    in_edge_weights = []
    con_edge_weights = []

    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            in_edge_weights.append(hedge2noderank[hidx][vorder])
            
    for vidx, node in enumerate(node2hedge):
        for horder, h in enumerate(node):
            data_dict[('edge', 'con', 'node')].append((h, vidx))
            con_edge_weights.append(node2hedgerank[vidx][horder])
    
    # print(in_edge_weights[0])
    in_edge_weights = torch.Tensor(in_edge_weights)
    con_edge_weights = torch.Tensor(con_edge_weights)
    # print(in_edge_weights.shape) -> (# edges, # rank dim)

    g = dgl.heterograph(data_dict)
    g['in'].edata['weight'] = in_edge_weights
    g['con'].edata['weight'] = con_edge_weights
#     for i in range(len(in_edge_weights[0])):
#         g['in'].edata['weight_' + str(i)] = in_edge_weights[:,i] # torch!
#         g['con'].edata['weight_' + str(i)] = con_edge_weights[:,i]
    
    return g
'''

class CustomMultiLayerNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, probname, replace=False, return_eids=False):
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts
        self.probname = probname
        self.replace = replace

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        frontier = sampling.sample_neighbors(g, seed_nodes, fanout, prob=self.probname, replace=self.replace)
        
        return frontier

def gen_sampleweighted_DGLGraph(args, hedge2node, hedge2noderank, hedge2nodeweight, node2hedge, node2hedgerank, node2hedgeweight, device):
    data_dict = defaultdict(list)
    in_edge_weights = []
    con_edge_weights = []
    in_edge_sampleweights = []
    con_edge_sampleweights = []

    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            in_edge_weights.append(hedge2noderank[hidx][vorder])
            in_edge_sampleweights.append(hedge2nodeweight[hidx][vorder])
            
    for vidx, node in enumerate(node2hedge):
        for horder, h in enumerate(node):
            data_dict[('edge', 'con', 'node')].append((h, vidx))
            con_edge_weights.append(node2hedgerank[vidx][horder])
            con_edge_sampleweights.append(node2hedgeweight[vidx][horder])
    
    in_edge_weights = torch.Tensor(in_edge_weights)
    con_edge_weights = torch.Tensor(con_edge_weights)
    in_edge_sampleweights = torch.Tensor(in_edge_sampleweights)
    con_edge_sampleweights = torch.Tensor(con_edge_sampleweights)

    g = dgl.heterograph(data_dict)
    g['in'].edata['weight'] = in_edge_weights
    g['con'].edata['weight'] = con_edge_weights
    g['in'].edata['sampleweight'] = in_edge_sampleweights
    g['con'].edata['sampleweight'] = con_edge_sampleweights
    
    return g

