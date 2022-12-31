from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, coo_matrix, identity
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI_SCORE
import numpy as np
import argparse
from scipy.sparse import diags
from sklearn import metrics
import os
import math

import hypernetx as hnx
import hypernetx.algorithms.hypergraph_modularity as hmod
import networkx as nx
from hypernetx import Entity
from pprint import pprint

from sklearn.cluster import SpectralClustering
def CHC(R, n_clusters):
    numE, numV = R.shape
    degE = R.sum(axis=1) ** (-0.5)
    degV = R.sum(axis=0) ** (-1)
    
    tmp = np.matmul(np.diag(degE), R)
    tmp = np.matmul(tmp, np.diag(degV))
    tmp = np.matmul(tmp, R.T)
    tmp = np.matmul(tmp, np.diag(degE))
    #lap = np.identity(numE) - tmp
    lap = tmp
    
    clustering = SpectralClustering(n_clusters=n_clusters, random_state=0).fit(lap)
    clusters = clustering.labels_
    
    return clusters
    
from sklearn.cluster import KMeans 
def KM(R, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(R)
    clusters =  kmeans.labels_
    
    return clusters

from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer
def NMFCL(R, n_clusters):
    model = NMF(n_components=n_clusters, random_state=0)
    W = model.fit_transform(R) # |E| * n_components 
    
    transformer = Normalizer().fit(W)  # rowwise...!
    normW = transformer.transform(W)
    
    clusters = np.argmax(normW, axis=1)
    
    return clusters
    
def SBC(R, n_clusters):
    degE = R.sum(axis=1) ** (-0.5) # D1
    degV = R.sum(axis=0) ** (-0.5) # D2
    A_n = np.matmul(np.matmul(np.diag(degE),R), np.diag(degV))
    
    u, s, v = np.linalg.svd(A_n)
    
    l = math.ceil(math.log2(n_clusters))
    Z = np.matmul(np.diag(degE), u[:, :l])
    
    return KM(Z, n_clusters)

# ======================================================================================================================
parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--inputdir', type=str, default="../downstreamdata/")
parser.add_argument('--dataname', type=str, default="DBLP_cat")
parser.add_argument('--notexist_hedgename', action='store_true')
parser.add_argument('--predict_path', type=str, default="../train_results/")
parser.add_argument('--n_clusters', type=int, default=4)
parser.add_argument('--weights', type=str, default="2,1,2")
args = parser.parse_args()

if len(args.predict_path) > 0:
    args.predict_path += args.dataname + "/"

dataname = args.dataname
if args.notexist_hedgename is False:
    exist_hedgename = True
else:
    exist_hedgename = False
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
ednw = [float(w) for w in args.weights.split(",")]#[2.0, 1.0, 2.0]
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
input_test_name = "{}/{}/test_hindex_0.txt".format(args.inputdir, dataname)
input_valid_name = "{}/{}/valid_hindex_0.txt".format(args.inputdir, dataname)
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
WR = csr_matrix((datas, (rows, cols)), shape=(numhedges, numnodes)).toarray()

km_w_cls = KM(WR, args.n_clusters)
km_w = NMI_SCORE(hedge2cluster, km_w_cls)
nmf_w_cls = NMFCL(WR, args.n_clusters)
nmf_w = NMI_SCORE(hedge2cluster, nmf_w_cls)
sbc_w_cls = SBC(WR, args.n_clusters)
sbc_w = NMI_SCORE(hedge2cluster, sbc_w_cls)

# Weighted without edge-dependent node labels ----------------------------------------------------
rows, cols = WR.nonzero()
NWR = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(numhedges, numnodes)).toarray()

chc_cls = CHC(NWR, args.n_clusters)
chc = NMI_SCORE(hedge2cluster, chc_cls)


# Result ------------------------------------------------------------------------------------------
print("Data Name", args.dataname)
print("CHC: {}".format(chc))
print("KM: {}".format(km_w))
print("NMF: {}".format(nmf_w))
print("SBC: {}".format(sbc_w))
