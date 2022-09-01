import random
import os
from tqdm import tqdm
# import pdb
import time
import argparse


class HyperGraph:
    def __init__(self, dataname, exist_hedgename):
        self.dataname = dataname
        self.hedge2node = []
        self.node2hedge = []
        self.node_reindexing = {}
        self.node_orgindex = {}
        self.numhedges = 0
        self.numnodes = 0
        
        with open("../dataset/" + dataname + "/hypergraph.txt", "r") as f:
            for _hidx, line in enumerate(f.readlines()):
                tmp = line.split("\t")
                hidx = self.numhedges
                if args.exist_hedgename:
                    papercode = tmp[0][1:-1] # without '
                    papercode = papercode.rstrip()
                    tmp = tmp[1:]
                self.hedge2node.append([])
                for node in tmp:
                    node = int(node.rstrip())
                    if node not in self.node_reindexing:
                        node_reindex = self.numnodes
                        self.node_reindexing[node] = node_reindex
                        self.node_orgindex[node_reindex] = node 
                        self.numnodes += 1 
                        self.node2hedge.append([])
                    nodeindex = self.node_reindexing[node]
                    self.hedge2node[hidx].append(nodeindex)
                    self.node2hedge[nodeindex].append(hidx)
                self.numhedges += 1

        print("Number of Hyperedges : " + str(self.numhedges))
        print("Number of Nodes : " + str(self.numnodes))

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--dataset_name', default="DBLP2", type=str)
parser.add_argument('--exist_hedgename', action='store_true')
parser.add_argument('--k', default=10000, type=int)
parser.add_argument('--repeat_idx', default=0, type=int)
args = parser.parse_args()
data = HyperGraph(args.dataset_name, args.exist_hedgename)

# BFS
hset = []
htable = [len(data.hedge2node[h]) for h in range(data.numhedges)]
check_node = [False for _ in range(data.numnodes)]
while len(hset) < args.k:
    start_node = random.randint(1, data.numnodes - 1)
    if check_node[start_node] is False:
        # BFS
        queue = [start_node]
        check_node[start_node] = True
        while (len(queue) > 0):
            v = queue.pop(0)
            for h in data.node2hedge[v]:
                htable[h] -= 1
                if htable[h] == 0:
                    hset.append(h)
                    if len(hset) == args.k:
                        break
            if len(hset) == args.k:
                break
            for h in data.node2hedge[v]:
                for nv in data.hedge2node[h]:
                    if (v != nv) and (check_node[nv] is False):
                        check_node[nv] = True
                        queue.append(nv)
# print(len(hset))
if args.repeat_idx > 0:
    outputname = "../dataset/{}/sampled_hset_{}_{}.txt".format(args.dataset_name, args.k, args.repeat_idx) 
else:
    outputname = "../dataset/{}/sampled_hset_{}.txt".format(args.dataset_name, args.k)
with open(outputname, "w") as f:
    for h in hset:
        f.write(str(h) + "\n")

# with open("dataset/" + args.dataset_name + "/sampled_node_" + str(args.k) + ".txt", "w") as f:
#     for v in range(data.numnodes):
#         if check_node[v] is True:
#             f.write(str(v) + "\n")
            

            
            