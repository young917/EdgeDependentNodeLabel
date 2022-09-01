
from collections import defaultdict
import pandas as pd
import numpy as np
import argparse

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


node_features = defaultdict(dict) # re-indexing : {}
hedge_features = defaultdict(dict) #re-indexing : {}

hedge2index = {} # e.g. papercode -> index
hedgename = {} # index -> e.g. papercode
hedge2node = []
hedge2nodepos = []
hedge2rank = defaultdict(list) # "degree" -> [[], [], ...], "eigenvec" -> [[], [], ...]
numhedges = 0

node2hedge = []
node_reindexing = {}
node_orgindex = {}
node2rank = defaultdict(list)
numnodes = 0

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--dataname', default='DBLP2', type=str)
parser.add_argument('--k', default=10000, type=int)
parser.add_argument('--repeat_idx', default=0, type=int)
parser.add_argument('--exist_hedgename', action='store_true')
parser.add_argument('--given_node_feat', default="", type=str) # ["affiliation"]
parser.add_argument('--given_hedge_feat', default="", type=str) # ["conf", "year", "cs", "de", "se", "th"]
args = parser.parse_args()

hset = []
if args.k > 0:
    if args.repeat_idx > 0:
        inputname = "../dataset/{}/sampled_hset_{}_{}.txt".format(args.dataname, args.k, args.repeat_idx)
    else:
        inputname = "../dataset/{}/sampled_hset_{}.txt".format(args.dataname, args.k)
    with open(inputname, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            hset.append(int(line))
        
with open("../dataset/" + args.dataname + "/hypergraph.txt", "r") as f:
    for _hidx, line in enumerate(f.readlines()):
        if (args.k == 0) or ((args.k > 0) and (_hidx in hset)):
            tmp = line.split("\t")
            hidx = numhedges
            if args.exist_hedgename:
                papercode = tmp[0][1:-1] # without '
                papercode = papercode.rstrip()
                hedge2index[papercode] = hidx
                hedgename[hidx] = papercode
                tmp = tmp[1:]
            else:
                hedge2index[_hidx] = hidx
                hedgename[hidx] = _hidx
            hedge2node.append([])
            hedge2nodepos.append([])
            for node in tmp:
                node = int(node.rstrip())
                if node not in node_reindexing:
                    node_reindex = numnodes
                    node_reindexing[node] = node_reindex
                    node_orgindex[node_reindex] = node 
                    numnodes += 1 
                    node2hedge.append([])
                nodeindex = node_reindexing[node]
                hedge2node[hidx].append(nodeindex)
                node2hedge[nodeindex].append(hidx)
            numhedges += 1

assert len(hedge2nodepos) == numhedges
with open("../dataset/" + args.dataname + "/hypergraph_pos.txt", "r") as f:
    for _hidx, line in enumerate(f.readlines()):
        tmp = line.split("\t")
        if args.exist_hedgename:
            papercode = tmp[0][1:-1] # without ''
            if (papercode not in hedge2index):
                continue
            hidx = hedge2index[papercode]
            tmp = tmp[1:]
        else:
            if (_hidx not in hedge2index):
                continue
            hidx = hedge2index[_hidx]
        hedge2nodepos[hidx] = [int(i) for i in tmp]
        
        assert len(hedge2nodepos[hidx]) == len(hedge2node[hidx])

for h in range(numhedges):
    assert len(hedge2nodepos[h]) == len(hedge2node[h]), h
    assert len(hedge2nodepos[h]) > 0

vfeats = ["degree", "eigenvec", "kcore", "pagerank"]
for vfeat in vfeats:
    if args.repeat_idx > 0:
        inputname = "../dataset/{}/{}_nodecentrality_{}_{}.txt".format(args.dataname, vfeat, args.k, args.repeat_idx)
    else:
        inputname = "../dataset/{}/{}_nodecentrality_{}.txt".format(args.dataname, vfeat, args.k)
    with open(inputname, "r") as f:
        _ = f.readline()
        for line in f.readlines():
            line = line.rstrip()
            orgindex, centrality = line.split("\t")
            orgindex, centrality = int(orgindex), float(centrality)
            if orgindex in node_reindexing:
                reindex = node_reindexing[orgindex]
                node_features[reindex][vfeat] = centrality
# affiliation
if len(args.given_node_feat) > 0 and args.exist_hedgename:
    target_cols = args.given_node_feat.split(",") 
    with open("../dataset/" + args.dataname + "/nodeinfo.txt", "r", encoding="utf-8") as f:
        columns = f.readline().rstrip()
        columns = columns.split("\t")
        for line in f.readlines():
            line = line.rstrip()
            tmp = line.split("\t")
            nodeindex = int(tmp[0])
            if nodeindex not in node_reindexing:
                # not include in incidence matrix
                continue
            node_reindex = node_reindexing[nodeindex]
            for i, col in enumerate(columns):
                if col in target_cols:
                    node_features[node_reindex][col] = tmp[i]
                
# make vfeat rank
# vrank_feats = ["degree_rank", "eigenvec_rank", "pagerank_rank", "kcore_rank"]
for vfeat in vfeats:
    for h in range(len(hedge2node)):
        vfeat_list = []
        for idx, v in enumerate(hedge2node[h]):
            vfeat_list.append(node_features[v][vfeat])
        vfeat_rank = make_ranking(vfeat_list)
        vfeat_rank = [vr / len(hedge2node[h]) for vr in vfeat_rank]
        # vfeat_list = sorted(vfeat_list, reverse=True)
        # vfeat_rank = [0 for _ in range(len(vfeat_list))]
        # prev_val = None
        # cur_rank = 0
        # for (val, idx) in vfeat_list:
        #     if prev_val is None:
        #         prev_val = val
        #     elif prev_val > val:
        #         cur_rank += 1
        #         prev_val = val
        #     vfeat_rank[idx] = cur_rank
        # vfeat_rank = np.argsort(vfeat_list).tolist()
        hedge2rank[vfeat + "_vrank"].append(vfeat_rank) # hyperedge order and vorder in hedge2node[h]
        
    for vidx, node in enumerate(node2hedge):
        ranks = []
        for hidx in node:
            for vorder,_v in enumerate(hedge2node[hidx]):
                if _v == vidx:
                    ranks.append(hedge2rank[vfeat + "_vrank"][hidx][vorder])
                    break
        node2rank[vfeat + "_erank"].append(ranks) # vertex order and horder in node2hedge[vidx]
    
'''
types = ["avg", "max", "min", "sum"]
for vfeat in vfeats:
    for t in types:
        with open("../dataset/" + args.dataname + "/" + vfeat + "_" + t + "_hfeatcentrality_" + str(args.k) + ".txt", "r") as f:
            for _hidx, line in enumerate(f.readlines()):
                line = line.rstrip()
                name, feat = line.split("\t")
                if args.exist_hedgename is False:
                    name = int(name)

                if name not in hedge2index:
                    continue
                hindex = hedge2index[name]
                feat = float(feat)
                hedge_features[hindex][vfeat + "_" + t] = feat
'''
# conference name?
if len(args.given_hedge_feat) > 0 and args.exist_hedgename:
    target_cols = args.given_hedge_feat.split(",")
    with open("../dataset/" + args.dataname + "/hyperedgeinfo.txt", "r", encoding="utf-8") as f:
        columns = f.readline().rstrip()
        columns = columns.split("\t")
        for _hidx, line in enumerate(f.readlines()):
            line = line.rstrip()
            tmp = line.split("\t")
            papercode = tmp[0][1:-1] # remove ''
            if papercode not in hedge2index:
                # not include in incidence matrix
                continue
            index = hedge2index[papercode]
            for i, col in enumerate(columns):
                if col in target_cols:
                    if col == "conf":
                        hedge_features[index][col] = tmp[i]
                    else:
                        hedge_features[index][col] = int(tmp[i])

vfeat_names = list(node_features[0].keys())
vfeatrank_names = list(hedge2rank.keys())
efeatrank_names = list(node2rank.keys())
# efeat_names = list(hedge_features[0].keys())
# print(vfeat_names)
# print(efeat_names)
if args.repeat_idx > 0:
    outputname = "../dataset/{}/whole_data_{}_{}.txt".format(args.dataname, args.k, args.repeat_idx)
else:
    outputname = "../dataset/{}/whole_data_{}.txt".format(args.dataname, args.k)
with open(outputname, "w", encoding='utf-8') as f:
    f.write(",".join(["node index", "node reindex", "hedgename"] + vfeat_names + vfeatrank_names + efeatrank_names + ["pos"]) + "\n")
    for h in range(numhedges):
        # paper_code = hedgename[h]
        assert len(hedge2node[h]) == len(hedge2nodepos[h]), str(len(hedge2node[h])) + "_" + str(len(hedge2nodepos[h]))
        for order, v in enumerate(hedge2node[h]):
            orgindex = node_orgindex[v]
            pos = hedge2nodepos[h][order]
            
            line = [str(orgindex), str(v), str(hedgename[h])]
            for vfeat in vfeat_names:
                line.append(str(node_features[v][vfeat]))
            for vfeatrank in vfeatrank_names:
                line.append(str(hedge2rank[vfeatrank][h][order]))
            for efeatrank in efeatrank_names:
                for horder, hidx in enumerate(node2hedge[v]):
                    if hidx == h:
                        break
                line.append(str(node2rank[efeatrank][v][horder]))
            # for efeat in efeat_names:
            #     line.append(str(hedge_features[h][efeat]))
            line.append(str(pos))
            
            f.write(",".join(line) + "\n")
