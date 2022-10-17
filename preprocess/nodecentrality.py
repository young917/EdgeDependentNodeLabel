from collections import defaultdict
import argparse
import numpy as np
from scipy import sparse as sp
import networkx as nx
import os
from scipy import sparse

class HyperGraph:
    def __init__(self, dataname, k, sample_type, exist_hedgename):
        self.dataname = dataname
        self.k = k
        self.hedge2node = []
        self.node2hedge = []
        self.node_reindexing = {}
        self.node_orgindex = {}
        self.numhedges = 0
        self.numnodes = 0
        max_size = 0
        
        if k > 0 :
            sampled_hset = []
            if sample_type == "rw":
                inputname = "../dataset/{}/sampled_hset_{}.txt".format(self.dataname, self.k)
                with open(inputname, "r") as f:
                    for line in f.readlines():
                        line = line.rstrip()
                        sampled_hset.append(int(line))  
            elif sample_type == "random":
                with open("../dataset/" + self.dataname + "/sampled_hset_random" + str(k) + ".txt", "r") as f:
                    for line in f.readlines():
                        line = line.rstrip()
                        sampled_hset.append(int(line))
            print(sampled_hset[:10])
                        
        with open("../dataset/" + self.dataname + "/hypergraph.txt", "r") as f:
            for _hidx, line in enumerate(f.readlines()):
                if (k == 0) or ((k > 0) and (_hidx in sampled_hset)):
                    tmp = line.split("\t")
                    if (max_size < (len(tmp) - 1)):
                        max_size = len(tmp) - 1
                    if exist_hedgename:
                        papercode = tmp[0][1:-1] # without '
                        papercode = papercode.rstrip()
                        tmp = tmp[1:]
                    hidx = self.numhedges
                    self.hedge2node.append([])
                    for node in tmp:
                        node = int(node.rstrip())
                        if node not in self.node_reindexing:
                            node_reindex = self.numnodes
                            self.node_reindexing[node] = node_reindex
                            self.node_orgindex[node_reindex] = node
                            self.node2hedge.append([])
                            self.numnodes += 1
                        nodeindex = self.node_reindexing[node]
                        self.hedge2node[hidx].append(nodeindex)
                        self.node2hedge[nodeindex].append(hidx)
                    self.numhedges += 1
         
        print("Max Size = ", max_size)
        print("Number of Hyperedges : " + str(self.numhedges))
        print("Number of Nodes : " + str(self.numnodes))

    def construct_weighted_clique(self):
        tmp_dict = defaultdict(int)
        values = []
        rows = []
        cols = []

        for v in range(self.numnodes):
            for h in self.node2hedge[v]:
                for nv in self.hedge2node[h]:
                    if v < nv:
                        key = str(v) + "," + str(nv)
                        tmp_dict[key] += 1 # num of common hedges

        for k, ch in tmp_dict.items():
            v, nv = k.split(",")
            v, nv = int(v), int(nv)
            values.append(ch)
            rows.append(v)
            cols.append(nv)
            
            values.append(ch)
            rows.append(nv)
            cols.append(v)
            
        values = np.array(values)
        rows = np.array(rows)
        cols = np.array(cols)
        self.weighted_matrix = sp.coo_matrix( (values, (rows, cols)), shape=(self.numnodes, self.numnodes))


def cal_kcore(graph):
    node_centrality = {}
    pos = [-1 for _ in range(graph.numnodes)]
    vert = [-1 for _ in range(graph.numnodes)]
    check_hedge = [False for _ in range(graph.numhedges)]

    node_degree = []
    md = 0
    for vidx in range(graph.numnodes):
        deg = len(graph.node2hedge[vidx])
        node_degree.append(deg)
        node_centrality[vidx] = deg
        if md < deg:
            md = deg
    bin = [0 for _ in range(md + 1)]
    for v in range(graph.numnodes):
        vdeg = node_degree[v]
        bin[vdeg] += 1

    start = 0
    for d in range(md + 1):
        num = bin[d]
        bin[d] = start
        start += num

    for v in range(graph.numnodes):
        pos[v] = bin[node_degree[v]]
        vert[pos[v]] = v
        bin[node_degree[v]] += 1

    for d in range(md, 0, -1):
        bin[d] = bin[d-1]
    bin[0] = 0

    previous = -1
    for i in range(graph.numnodes):
        v = vert[i]
        assert previous <= node_centrality[v]
        previous = node_centrality[v]

    for i in range(graph.numnodes):
        v = vert[i]
        vdeg = node_degree[v]
        for hidx in range(vdeg):
            h = graph.node2hedge[v][hidx]
            if check_hedge[h] is False:
                hsize = len(graph.hedge2node[h])
                for nvidx in range(hsize):
                    nv = graph.hedge2node[h][nvidx]
                    if node_centrality[nv] > node_centrality[v]:
                        dnv = node_centrality[nv]
                        pnv = pos[nv]
                        pw = bin[dnv]
                        w = vert[pw]
                        if nv != w:
                            pos[nv] = pw
                            pos[w] = pnv
                            vert[pnv] = w
                            vert[pw] = nv
                        bin[dnv] += 1
                        node_centrality[nv] -= 1
                check_hedge[h] = True

    return node_centrality

def cal_degree(graph):
    node_centrality = {}
    for vidx in range(graph.numnodes):
        deg = len(graph.node2hedge[vidx])
        node_centrality[vidx] = deg
    return node_centrality

def cal_pagerank(graph):
    graph.construct_weighted_clique()
    nx_graph = nx.convert_matrix.from_scipy_sparse_matrix(graph.weighted_matrix)

    return nx.algorithms.link_analysis.pagerank_alg.pagerank(nx_graph)

def cal_eigenvector(graph):
    graph.construct_weighted_clique()
    nx_graph = nx.convert_matrix.from_scipy_sparse_matrix(graph.weighted_matrix)

    return nx.eigenvector_centrality(nx_graph)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', required=False, default="degree")
    parser.add_argument('--dataname', required=False, default="DBLP2")
    parser.add_argument('--exist_hedgename', action='store_true')
    parser.add_argument('--k', required=False, type=int, default=0, help="size of sampled hypergraph from dataset")
    parser.add_argument('--sampletype', required=False, type=str, default="rw", help="type of sampled hypergraph from dataset")
    args = parser.parse_args()

    graph = HyperGraph(args.dataname, args.k, args.sampletype, args.exist_hedgename)
    if args.algo == "degree":
        node_centrality = cal_degree(graph)
    elif args.algo == "kcore":
        node_centrality = cal_kcore(graph)
    elif args.algo == "pagerank":
        node_centrality = cal_pagerank(graph)
    elif args.algo == "eigenvec":
        node_centrality = cal_eigenvector(graph)

    outputdir = "../dataset/" + args.dataname + "/"
    if os.path.isdir(outputdir) is False:
        os.makedirs(outputdir)
    outputname = outputdir + "{}_nodecentrality_{}.txt".format(args.algo, args.k)
    with open(outputname, "w") as f:
        f.write("node\t" + args.algo + "\n")
        for v in node_centrality:
            node_orgindex = graph.node_orgindex[v]
            f.write(str(node_orgindex) + "\t" + str(node_centrality[v]) + "\n")
            