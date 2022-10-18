from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, coo_matrix, identity
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI_SCORE
import numpy as np
import argparse
from scipy.sparse import diags
from sklearn import metrics
import os

import hypernetx as hnx
import hypernetx.algorithms.hypergraph_modularity as hmod
import networkx as nx
from hypernetx import Entity
from pprint import pprint

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--inputdir', type=str)
parser.add_argument('--dataname', type=str)
parser.add_argument('--exist_hedgename', action='store_true')
parser.add_argument('--predict_path', type=str, default="")
args = parser.parse_args()

dataname = args.dataname
exist_hedgename = args.exist_hedgename
numhedges = 0
numnodes = 0
hedgeindex = {}
hedge2node = []
hedge2nodepos = []
hedge2nodepos_predict = []
node_reindexing = {}
node2hedge = []

# Read Data ======================================================================================================================
# hypergraph --------------------------------------------------------------------
with open(args.inputdir + dataname + "/hypergraph.txt", "r") as f:
    for _hidx, line in enumerate(f.readlines()):
            tmp = line.split("\t")
            hidx = numhedges
            numhedges += 1
            if exist_hedgename:
                papercode = tmp[0][1:-1] # without '
                papercode = papercode.rstrip()
                hedgeindex[papercode] = hidx
                tmp = tmp[1:]
            else:
                hedgeindex[_hidx] = hidx
            hedge2node.append([])
            hedge2nodepos.append([])
            hedge2nodepos_predict.append([])
            for node in tmp:
                node = int(node.rstrip())
                if node not in node_reindexing:
                    node_reindex = numnodes
                    numnodes += 1 
                    node_reindexing[node] = node_reindex
                    node2hedge.append([])
                nodeindex = node_reindexing[node]
                hedge2node[hidx].append(nodeindex)
                node2hedge[nodeindex].append(hidx)
# weighting by edge-dependent node label --------------------------------------------------------------------
ednw = [2.0, 1.0, 2.0]
with open(args.inputdir + dataname + "/hypergraph_pos.txt", "r") as f:
    for _hidx, line in enumerate(f.readlines()):
        tmp = line.split("\t")
        if exist_hedgename:
            papercode = tmp[0][1:-1] # without ''
            if (papercode not in hedgeindex):
                continue
            hidx = hedgeindex[papercode]
            tmp = tmp[1:]
        else:
            if (_hidx not in hedgeindex):
                continue
            hidx = hedgeindex[_hidx]
        positions = [ednw[int(i)] for i in tmp]
        for nodepos in positions:
            hedge2nodepos[hidx].append(nodepos)
            
if len(args.predict_path) > 0:
    with open(args.predict_path + "/prediction.txt", "r") as f:
        for _hidx, line in enumerate(f.readlines()):
            tmp = line.split("\t")
            hidx = _hidx
            positions = [ednw[int(i)] for i in tmp]
            for nodepos in positions:
                hedge2nodepos_predict[hidx].append(nodepos)
            
# train-test split --------------------------------------------------------------------         
input_test_name = "{}/{}/test_hindex.txt".format(args.inputdir, dataname)
input_valid_name = "{}/{}/valid_hindex.txt".format(args.inputdir, dataname)
valid_index = []
test_index = []
train_index = []
with open(input_valid_name, "r") as f:
    for line in f.readlines():
        name = line.rstrip()
        if exist_hedgename is False:
            name = int(name)
        index = hedgeindex[name]
        valid_index.append(index)
with open(input_test_name, "r") as f:
    for line in f.readlines():
        name = line.rstrip()
        if exist_hedgename is False:
            name = int(name)
        index = hedgeindex[name]
        test_index.append(index)
train_index = set(range(numhedges)) - (set(valid_index + test_index))

print("# Hyperedges:", numhedges)
print("# Nodes:", numnodes)
nnz = 0
for hedge in hedge2node:
    for node in hedge:
        nnz += 1
nnz /= (numhedges * numnodes)
print("NNZ: %.4f" % nnz)
                
# Clustering ============================================================================================
# answer cluster -------------------------------------------
hedge2cluster = np.zeros(numhedges)
with open(args.inputdir + dataname + "/hyperedge_cluster.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        if exist_hedgename:
            hedgename, cl_str = line.rstrip().split("\t")
            h_index = hedgeindex[hedgename]
            cl = int(float(cl_str))
        else:
            h_index = hedgeindex[i]
            cl = int(line.rstrip())
        hedge2cluster[h_index] = cl
        
# Weighted with Ground-Truth edge-dependent node labels --------------------------------------------
rows, cols, datas = [], [], []
for h in range(numhedges):
    for vi, w in enumerate(hedge2nodepos[h]):
        v = hedge2node[h][vi]
        rows.append(h)
        cols.append(v)
        datas.append(w)
weights_from_gt = csr_matrix((datas, (rows, cols)), shape=(numhedges, numnodes))
# Weighted with Predicted edge-dependent node labels ------------------------------------
rows, cols, datas = [], [], []
for h in range(numhedges):
    for vi, w in enumerate(hedge2nodepos_predict[h]):
        v = hedge2node[h][vi]
        rows.append(h)
        cols.append(v)
        datas.append(w)
weights_from_pred = csr_matrix((datas, (rows, cols)), shape=(numhedges, numnodes))
# Weighted without edge-dependent node labels -----------------------------------------
rows, cols = weights_from_gt.nonzero()
weights_from_nolab = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(numhedges, numnodes))

def get_cluster(weights, k, weightflag=True):
    mat = weights.tocoo()
    cols = mat.col
    rows = mat.row # target
    data = np.array([cols, rows]).T
    if weightflag:
        w = mat.data
        h = hnx.Hypergraph(hnx.StaticEntitySet(data=data, weights=w), weights=w)
#         assert h.is_connected()
        clusterlist = hnx.spec_clus(h, k, weights=True)
    else:
        h = hnx.Hypergraph(hnx.StaticEntitySet(data=data))
#         assert h.is_connected()
        clusterlist = hnx.spec_clus(h, k, weights=False)
    _clusters = {}
    for i in clusterlist:
        for v in clusterlist[i]:
            _clusters[v] = i
    assert len(_clusters.keys()) == weights.shape[0]
    clusters = [_clusters[i] for i in range(weights.shape[0])]
    del mat
    del h
    del clusterlist
    
    return clusters

gt_clusters = get_cluster(weights_from_gt, args.k, weightflag=True)
gt_score = NMI_SCORE(hedge2cluster, true_clusters)
predict_clusters = get_cluster(weights_from_pred, args.k, weightflag=True)
predict_score = NMI_SCORE(hedge2cluster, predict_clusters)
nolab_clusters = get_cluster(weights_from_nolab, args.k, weightflag=False)
nolab_score = NMI_SCORE(hedge2cluster, unif_clusters)
rand_clusters = [np.random.randint(args.k) for h in range(numhedges)]
rand_score = NMI_SCORE(hedge2cluster, rand_clusters)

print("Data Name", args.dataname)
print("Hypergraph w/ GroundTruth\t\t{}\n".format(gt_score))
print("Hypergraph w/ WHATsNet\t\t{}\n".format(predict_score))
print("Hypergraph w/o Labels\t\t{}\n".format(nolab_score))
print("Random\t\t{}\n".format(rand_score))
print()
