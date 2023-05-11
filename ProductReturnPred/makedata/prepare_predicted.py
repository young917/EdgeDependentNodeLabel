import numpy as np
import matplotlib.pylab as plt
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from scipy.stats import truncnorm 
import pickle
from collections import defaultdict

import argparse
parser = argparse.ArgumentParser(description='Argparse Tutorial')
# parser.add_argument('--predict_path', default="prediction")
parser.add_argument('--outputname', default="our_0") # "h_WHATsNet_mat"
args = parser.parse_args()



# n : nodes, m : hyperedges
with open("../data/h_mat.pkl", 'rb') as f:
    h = pickle.load(f)
n, m = h.shape

# make indexing dictionary!
hedge2node = []
hedge2nodepos = []
hedge2index = {} # h -> h_prime
index2hedge = {} # h_prime -> h

node2hedge = []
node2index = {} # v -> v_prime
index2node = {} # v_prime -> v

numhedges = 0
numnodes = 0

h_coo = h.tocoo()
for vidx, hidx, vw in zip(h_coo.row, h_coo.col, h_coo.data):
    if hidx not in hedge2index:
        hedge2index[hidx] = numhedges
        index2hedge[numhedges] = hidx
        numhedges += 1
        hedge2node.append([])
        hedge2nodepos.append([])
    
    if vidx not in node2index:
        node2index[vidx] = numnodes
        index2node[numnodes] = vidx
        numnodes += 1
        node2hedge.append([])
    
    re_hidx = hedge2index[hidx]
    re_vidx = node2index[vidx]
    
    hedge2node[re_hidx].append(re_vidx)
    hedge2nodepos[re_hidx].append(vw)
    node2hedge[re_vidx].append(re_hidx)

print(numnodes, numhedges)

# check with "data/h_mat.pkl" and "hypergraph.txt"
# == "hedge2node" and "hypergraph.txt"

graph_fname = "../../downstreamdata/Etail/hypergraph.txt"

with open(graph_fname, "r") as gf:
    for h_prime, gline in enumerate(gf.readlines()):
        nodes1 = [int(v) for v in gline.rstrip().split("\t")]
        nodes2 = hedge2node[h_prime]
        
        assert sorted(nodes1) == sorted(nodes2)

# predict_path = "../../train_results/Etail/prediction.txt"
nodes_for_h = []
hedges_for_h = []
data = []

hidx = 0
with open("../../train_results/Etail/prediction_{}.txt".format(args.outputname), "r") as pf, open(graph_fname, "r") as gf:
    for pline, gline in zip(pf.readlines(), gf.readlines()):
        _weights = [int(float(p)) for p in pline.rstrip().split("\t")]
        _nodes = [int(v) for v in gline.rstrip().split("\t")]
        assert len(_weights) == len(_nodes)
        
        for v, w in zip(_nodes, _weights):
            nodes_for_h.append(index2node[v])
            hedges_for_h.append(index2hedge[hidx])
            data.append(w)
            
        hidx += 1
        
h_our = csr_matrix((data, (nodes_for_h, hedges_for_h)), shape=(n, m))
# same nonzero!
check_dict = defaultdict(int)
vs_our, hs_our = h_our.nonzero()
for _v, _h in zip(vs_our, hs_our):
    key = str(_v) + "_" + str(_h)
    check_dict[key] = 1
    
vs_ori, hs_ori = h.nonzero()
assert vs_our.shape[0] == vs_ori.shape[0]
for _v, _h in zip(vs_ori, hs_ori):
    key = str(_v) + "_" + str(_h)
    assert check_dict[key] == 1

outputdir = "../data/"
outputpath = outputdir + "h_{}_mat.pkl".format(args.outputname)
with open(outputpath, 'wb') as f:
    pickle.dump(h_our, f)