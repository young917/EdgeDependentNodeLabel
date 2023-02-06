from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, coo_matrix, identity
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI_SCORE
import numpy as np
import argparse
from scipy.sparse import diags
from sklearn import metrics
import os
import tqdm

import hypernetx as hnx
import hypernetx.algorithms.hypergraph_modularity as hmod
import networkx as nx
from hypernetx import Entity
from pprint import pprint

# code referenced from https://github.com/pnnl/HyperNetX/blob/master/tutorials/Tutorial%2011%20-%20Laplacians%20and%20Clustering.ipynb

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--inputdir', type=str, default="../dataset/Downstream/AMiner/")
parser.add_argument('--dataname', type=str, default="AMiner")
parser.add_argument('--predict_path', type=str, default="../train_results/AMiner/")
parser.add_argument('--n_cluster', type=int, default=4)
args = parser.parse_args()

numhedges = 0
numnodes = 0
hedge2node = []
hedge2nodepos = []
hedge2nodepos_predict = []
node2hedge = []

sampled_paperid = set()
paperid2hindex = {}
hindex2paperid = {}

authorid2vindex = {}
vindex2authorid = {}

category_indexing = {}

# Read Data ======================================================================================================================
# hypergraph --------------------------------------------------------------------
with open(args.inputdir + "sampled_paperid_10000.txt", "r") as f:
    for line in f.readlines():
        paperid = line.rstrip()
        sampled_paperid.add(paperid)

with open(args.inputdir + "hypergraph.txt", "r") as f, open(args.inputdir + "hypergraph_pos.txt", "r") as pf:
    for line, pline in zip(f.readlines(), pf.readlines()):
        nodes = line.rstrip().split("\t")
        node_poses = pline.rstrip().split("\t")[1:]
        paperid = nodes[0]
        nodes = nodes[1:]
        if len(nodes) == 1:
            continue
        elif paperid not in sampled_paperid:
            continue
        hindex = numhedges
        paperid2hindex[paperid] = hindex
        hindex2paperid[hindex] = paperid
        numhedges += 1
        hedge2node.append([])
        hedge2nodepos.append([])
        hedge2nodepos_predict.append([])
        
        for authorid, _vpos in zip(nodes, node_poses):
            if authorid not in authorid2vindex:
                vindex = numnodes
                authorid2vindex[authorid] = vindex
                vindex2authorid[vindex] = authorid
                numnodes += 1
                node2hedge.append([])
            vindex = authorid2vindex[authorid]
            if int(_vpos) == 1:
                vpos = 2.0
            elif int(_vpos) == len(nodes):
                vpos = 2.0
            else:
                vpos = 1.0
            hedge2node[hindex].append(int(vindex))
            hedge2nodepos[hindex].append(float(vpos))
            node2hedge[vindex].append(hindex)
            
if len(args.predict_path) > 0:
    with open(args.predict_path + "/prediction.txt", "r") as f:
        for _hidx, line in enumerate(f.readlines()):
            tmp = line.split("\t")
            hidx = _hidx
            # positions = [ednw[int(i)] for i in tmp]
            positions = []
            for i in tmp:
                if int(i) == 0:
                    positions.append(2.0)
                elif int(i) == 1:
                    positions.append(1.0)
                elif int(i) == 2:
                    positions.append(2.0)
            for nodepos in positions:
                hedge2nodepos_predict[hidx].append(nodepos)
            assert len(hedge2nodepos[hidx]) == len(hedge2nodepos_predict[hidx])

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

with open(args.inputdir + "sampled_paperid.txt", "r") as f:
    for line in f.readlines():
        paperid, category_name = line.rstrip().split("\t")
        if paperid in paperid2hindex:
            hindex = paperid2hindex[paperid]
            if category_name not in category_indexing:
                category_indexing[category_name] = len(category_indexing)
            category = category_indexing[category_name]
            hedge2cluster[hindex] = category
print(len(category_indexing))
            
# Weighted with Ground-Truth edge-dependent node labels --------------------------------------------
rows, cols, datas = [], [], []
for h in range(numhedges):
    for vi, w in enumerate(hedge2nodepos[h]):
        v = hedge2node[h][vi]
        rows.append(h)
        cols.append(v)
        datas.append(w)
# print(datas)
weights_from_gt = csr_matrix((datas, (rows, cols)), shape=(numhedges, numnodes))
print("Weight from GT")
# Weighted with Predicted edge-dependent node labels ------------------------------------
if len(args.predict_path) > 0:
    rows, cols, datas = [], [], []
    for h in range(numhedges):
        for vi, w in enumerate(hedge2nodepos_predict[h]):
            v = hedge2node[h][vi]
            rows.append(h)
            cols.append(v)
            datas.append(w)
    weights_from_pred = csr_matrix((datas, (rows, cols)), shape=(numhedges, numnodes))
    print("Weight from Prediction")
# Weighted without edge-dependent node labels -----------------------------------------
rows, cols = weights_from_gt.nonzero()
weights_from_nolab = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(numhedges, numnodes))
print("Weight from w/o Label")

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

gt_clusters = get_cluster(weights_from_gt, args.n_cluster, weightflag=True)
gt_score = NMI_SCORE(hedge2cluster, gt_clusters)
print("Clusters from GT")
if len(args.predict_path) > 0:
    predict_clusters = get_cluster(weights_from_pred, args.n_cluster, weightflag=True)
    predict_score = NMI_SCORE(hedge2cluster, predict_clusters)
    print("Clusters from Prediction")
nolab_clusters = get_cluster(weights_from_nolab, args.n_cluster, weightflag=False)
nolab_score = NMI_SCORE(hedge2cluster, nolab_clusters)
print("Clusters from w/o Label")
rand_clusters = [np.random.randint(args.n_cluster) for h in range(numhedges)]
rand_score = NMI_SCORE(hedge2cluster, rand_clusters)

print("Data Name", args.dataname)
print("Hypergraph w/ GroundTruth\t\t{}\n".format(gt_score))
if len(args.predict_path) > 0:
    print("Hypergraph w/ WHATsNet\t\t{}\n".format(predict_score))
print("Hypergraph w/o Labels\t\t{}\n".format(nolab_score))
print("Random\t\t{}\n".format(rand_score))
print()
