from collections import defaultdict
import argparse
import numpy as np
from scipy import sparse as sp
import networkx as nx
import os
from scipy import sparse

class HyperGraph:
    def __init__(self, dataname, exist_hedgename):
        self.dataname = dataname
        self.hedge2node = []
        self.node2hedge = []
        self.node_reindexing = {}
        self.node_orgindex = {}
        self.numhedges = 0
        self.numnodes = 0
        self.max_size = 0
        
        sampled_hset = []
        with open("../dataset/" + self.dataname + "/hypergraph.txt", "r") as f:
            for _hidx, line in enumerate(f.readlines()):
                tmp = line.split("\t")
                if (self.max_size < len(tmp)):
                    self.max_size = len(tmp)
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
         
        print("Max Size = ", self.max_size)
        print("Number of Hyperedges : " + str(self.numhedges))
        print("Number of Nodes : " + str(self.numnodes))

    def construct_line_graph(self):
        tmp_dict = defaultdict(int)
        values = []
        rows = []
        cols = []

        for h in range(self.numhedges):
            for v in self.hedge2node[h]:
                for nh in self.node2hedge[v]:
                    if h < nh:
                        key = str(h) + "," + str(nh)
                        tmp_dict[key] += 1

        for k, ch in tmp_dict.items():
            h, nh = k.split(",")
            h, nh = int(h), int(nh)
            values.append(1) # ch
            rows.append(h)
            cols.append(nh)
            
            values.append(1) # ch
            rows.append(nh)
            cols.append(h)
            
        values = np.array(values)
        rows = np.array(rows)
        cols = np.array(cols)
        line_matrix = sp.coo_matrix( (values, (rows, cols)), shape=(self.numhedges, self.numhedges))

        nx_graph = nx.convert_matrix.from_scipy_sparse_matrix(line_matrix)
        he_centrality = nx.eigenvector_centrality(nx_graph)

        return he_centrality


def cal_vector_eig(graph):
    he_centrality = graph.construct_line_graph()
    assert len(he_centrality.keys()) == graph.numhedges
    node_centrality = {}
    for v in range(graph.numnodes):
        node_centrality[v] = [0 for _ in range(graph.max_size - 1)]
    for v in range(graph.numnodes):
        for h in graph.node2hedge[v]:
            hsize = len(graph.hedge2node[h])
            if hsize < 2:
                continue
            node_centrality[v][hsize-2] = node_centrality[v][hsize-2] + (he_centrality[h] / hsize)

    return node_centrality, he_centrality


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', required=False, default="StackOverflowBiology")
    parser.add_argument('--exist_hedgename', action='store_true')
    args = parser.parse_args()

    graph = HyperGraph(args.dataname, args.exist_hedgename)
    node_centrality, hedge_centrality = cal_vector_eig(graph)
    
    outputdir = "../dataset/" + args.dataname + "/"
    outputname = outputdir + "ve_nodecentrality_0.txt"
    with open(outputname, "w") as f:
        f.write("node\tve\n")
        for v in node_centrality:
            node_orgindex = graph.node_orgindex[v]
            cent_str = [str(vcent) for vcent in node_centrality[v]]
            line = "\t".join([str(node_orgindex)] + cent_str)
            f.write(line + "\n")
    outputname = outputdir + "ve_hedgecentrality_0.txt"
    with open(outputname, "w") as f:
        f.write("hedge\tve\n")
        for h in hedge_centrality:
            f.write(str(h) + "\t" + str(hedge_centrality[h]) + "\n")

    # postprocess
    nodecentrality = {}
    with open(outputdir + "ve_nodecentrality_0.txt", "r") as f:
        f.readline()
        for line in f.readlines():
            tmp = line.rstrip().split("\t")
            vindex = int(tmp[0])
            nodecentrality[vindex] = tmp[1:]
    dim = len(nodecentrality[0])
    for d in range(dim):
        with open(outputdir + f"ve{d}_nodecentrality_0.txt", "w") as of:
            of.write("Node\tCentrality\n")
            for orgindex in nodecentrality.keys():
                of.write(str(orgindex) + "\t" + str(nodecentrality[orgindex][d]) + "\n")
            