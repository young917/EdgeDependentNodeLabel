import csv
from datetime import datetime
import time
import random
import pickle
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import sparse
from scipy.stats import weightedtau, kendalltau
from scipy.stats import norm
from scipy.linalg import null_space
from sklearn.preprocessing import normalize
import pandas as pd

from pprint import pprint
from copy import deepcopy
from collections import defaultdict

import trueskill

# randomwalk function ----------------------------------------------------------------------------------------------------
def compute_pr(P, r, n, eps=1e-8):
    x = np.ones(n) / n*1.0
    flag = True
    t=0
    while flag:
        x_new = (1-r)*P*x
        x_new = x_new + np.ones(n) * r / n
        diff = np.linalg.norm(x_new - x)
        if np.linalg.norm(x_new - x,ord=1) < eps and t > 100:
            flag = False
        t=t+1
        x = x_new
    return x

# prepare data-------------------------------------------------------------------------------------------------------------
inputdir = "../dataset/Downstream/AMiner/"
authors = set()
sampled_paperid = set()

papers = [] # (author, pos)
paperid2hindex = {}
hindex2paperid = {}
numhedges = 0

authorid2vindex = {}
vindex2authorid = {}
numnodes = 0

vindex2rank = {}
hindex2cite = {}
category_indexing = {}
hindex2category = {}

with open(inputdir + "sampled_paperid_10000.txt", "r") as f:
    for line in f.readlines():
        paperid = line.rstrip().split("\t")[0]
        sampled_paperid.add(paperid)

with open(inputdir + "hypergraph.txt", "r") as f, open(inputdir + "hypergraph_pos.txt", "r") as pf:
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
        
        paper = []
        poses = []
        for authorid, _vpos in zip(nodes, node_poses):
            if authorid not in authorid2vindex:
                vindex = numnodes
                authorid2vindex[authorid] = vindex
                vindex2authorid[vindex] = authorid
                numnodes += 1
            vindex = authorid2vindex[authorid]
            if int(_vpos) == 1:
                vpos = 2
            elif int(_vpos) == len(nodes):
                vpos = 2
            else:
                vpos = 1
            paper.append(int(vindex))
            poses.append(int(vpos))
        papers.append((paper,poses))
        
with open(inputdir + "hypergraph_rank.txt", "r") as f:
    for line in f.readlines():
        authorid, vrank = line.rstrip().split("\t")
        if authorid in authorid2vindex:
            vindex = authorid2vindex[authorid]
            vindex2rank[vindex] = float(vrank)

with open(inputdir + "hypergraph_citation.txt", "r") as f:
    for line in f.readlines():
        paperid, citation = line.rstrip().split("\t")
        if paperid in paperid2hindex:
            hindex = paperid2hindex[paperid]
            hindex2cite[hindex] = int(citation)

# with open("sampled_paperid.txt", "r") as f:
#     for line in f.readlines():
#         paperid, category_name = line.rstrip().split("\t")
#         if paperid in paperid2hindex:
#             hindex = paperid2hindex[paperid]
#             if category_name not in category_indexing:
#                 category_indexing[category_name] = len(category_indexing)
#             category = category_indexing[category_name]
#             hindex2category = category

authors=list(range(numnodes)) # list of players

print("E:", len(papers))
print("V:", len(authors))

avg_size = 0
nnz = 0
for i in range(len(papers)):
    pi, score = papers[i]
    nnz += len(pi)
    avg_size += len(pi)
print("NNZ: ", nnz / (len(papers) * len(authors)))
print("AVG size", avg_size / len(papers))

# Weighted Hypergraph Ranking (GroundTruth) --------------------------------------------------------------------------------------
print("Groud Truth Weight Ranking ---------------")

pi_list = papers
universe = np.array(list(authors))
numhedges = len(pi_list)
numnodes = len(universe)
# first create these matrices
# R = |E| x |V|, R(e, v) = lambda_e(v)
# W = |V| x |E|, W(v, e) = w(e) 1(v in e)
    
m = len(pi_list) # number of hyperedges
n = len(universe) # number of items to be ranked 
R_rows, R_cols, R_data = [], [], []
W_rows, W_cols, W_data = [], [], []

for i in range(len(pi_list)):
    pi, scores = pi_list[i]
    if len(pi) > 1:   
        for j in range(len(pi)):
            v = pi[j]
            # v = np.where(universe == v)[0][0] #equivalent to universe.index(v) but for np arrays
            R_rows.append(i)
            R_cols.append(v)
            R_data.append(scores[j])
            
            W_rows.append(v)
            W_cols.append(i)
            W_data.append(hindex2cite[i] + 1)

R = sparse.csr_matrix((R_data, (R_rows, R_cols)), shape=(numhedges, numnodes))
R = normalize(R, norm='l1', axis=1)
W = sparse.csr_matrix((W_data, (W_rows, W_cols)), shape=(numnodes, numhedges))
Wnorm = normalize(W, norm='l1', axis=1)
# create prob trans matrices
P = Wnorm * R
P = P.T
# create rankings
r=0.40
rankings_hg = compute_pr(P, r, n, eps=1e-8).flatten()
del P

# G^H Ranking ---------------------------------------------------------------------------------------------------------------------
print("G^H Ranking ---------------")

import multiprocessing as mp

def pairwise_calculate(return_dict, row, pc, rownum):
    nonzero_indexes = row.nonzero()[0]
    _data = {}
    for uidx in range(len(nonzero_indexes)):
        for vidx in range(uidx + 1, len(nonzero_indexes)):
            u = nonzero_indexes[uidx]
            v  = nonzero_indexes[vidx]
            assert u < v
            key = str(u)+ "_" + str(v)
            if key not in _data:
                _data[key] = 0
            _data[key] += pc * row[u] * row[v]
    return_dict[rownum] = _data

E, V = R.shape
A_rows, A_cols, A_data = [], [], []
manager = mp.Manager()
return_dict = manager.dict()
n_cores = 8
# created pool running maximum n cores
pool = mp.Pool(n_cores)

for i in range(0, E):
    r = R.getrow(i).toarray()[0]
    nonzero_indexes = r.nonzero()[0]
    pc = hindex2cite[i]
    pool.apply_async(pairwise_calculate, args=(return_dict, r, pc, i))
    # Tell the pool that there are no more tasks to come and join
pool.close()
pool.join()
print("End MultiProcesssing . . .")

agg_dict = defaultdict(float)
for i in return_dict.keys():
    dic = return_dict[i]
    for str_key in dic.keys():
        agg_dict[str_key] += dic[str_key]
del return_dict

for str_key in agg_dict.keys():
    u,v = str_key.split("_")
    u,v = int(u), int(v)
    A_rows.append(u)
    A_cols.append(v)
    A_data.append(agg_dict[str_key])
    A_rows.append(v)
    A_cols.append(u)
    A_data.append(agg_dict[str_key])

A = sparse.csr_matrix((A_data, (A_rows, A_cols)), shape=(V,V))
P = normalize(A, norm='l1', axis=0)
P = P.T
# compute pagerank scores
r=0.40
rankings_gh = compute_pr(P, r, n, eps=1e-8).flatten()
del P

# MC3 ---------------------------------------------------------------------------------------------------------------------------
print("MC3 Ranking ---------------")

def row_execute(return_dict, row, scores, rownum):
    _data = {}
    winning = [0 for _ in range(len(row))]
    for uidx in range(len(row)):
        for vidx in range(uidx + 1, len(row)):
            u = row[uidx]
            v  = row[vidx]
            uscore = scores[uidx]
            vscore = scores[vidx]
            if uscore > vscore:
                winning[uidx] += 1
                # key = str(u)+ "_" + str(v)
                key = str(v) + "_" + str(u)
                _data[key] = 1/len(row)
            elif uscore < vscore:
                winning[vidx] += 1
                # key = str(v)+ "_" + str(u)
                key = str(u)+ "_" + str(v)
                _data[key] = 1/len(row)
    for uidx in range(len(row)):
        u = row[uidx]
        key = str(u) + "_" + str(u)
        # _data[key] = 1 - (winning[uidx] / len(row))
        _data[key] = winning[uidx] / len(row)
    return_dict[rownum] = _data

n_cores = 12
# created pool running maximum n cores
pool = mp.Pool(n_cores)
return_dict = manager.dict()
for i, (pi, scores) in enumerate(pi_list):
    pool.apply_async(row_execute, args=(return_dict, pi, scores, i))
pool.close()
pool.join()
print("End MultiProcesssing . . .")

deg = np.zeros(len(universe))
agg_dict = defaultdict(float)
for rowname in return_dict.keys():
    _data = return_dict[rowname]
    for str_key in _data.keys():
        u, v = str_key.split("_")
        u, v = int(u), int(v)
        if u == v:
            deg[u] += 1
        agg_dict[str_key] = agg_dict[str_key] + _data[str_key]
for str_key in agg_dict.keys():
    u, v = str_key.split("_")
    u, v = int(u), int(v) 
    if deg[u] > 0:
        agg_dict[str_key] = (agg_dict[str_key]) / deg[u]
    
for u in range(len(universe)):
    if deg[u] == 0:
        agg_dict[str(u) + "_" + str(u)] = 1
        
Pd_rows, Pd_cols, Pd_data = [], [], []
for str_key in agg_dict.keys():
    u, v = str_key.split("_")
    u, v = int(u), int(v)
    Pd_rows.append(u)
    Pd_cols.append(v)
    Pd_data.append(agg_dict[str_key])
Pd = sparse.csr_matrix((Pd_data, (Pd_rows, Pd_cols)), shape=(len(universe),len(universe)))
Pd = Pd.T
# create MC3 rankings
r=0.40
rankings_mc3 = compute_pr(Pd, r, len(universe), eps=1e-8).flatten()

# TrueSkill -----------------------------------------------------------------------------------------------------------------------
# simulate the change in TrueSkill ratings when a Free-For-All match is played
# print("TrueSkill Ranking ---------------")
# def play_match(match, ts_ranking):
#     p, s = match
#     cur_ranks = []
#     for player in p:
#         if player in ts_ranking:
#             cur_ranks.append([ts_ranking[player]])
#         else:
#             cur_ranks.append([trueskill.Rating()])
#     # lower rank = better player for trueskill.rate function, so we turn scores into -1*scores
#     match_res = trueskill.rate(cur_ranks, ranks=[-1*i for i in s])
#     for i in range(len(p)):
#         player = p[i]
#         ts_ranking[player] = match_res[i][0]
# trueskill_rankings={} # dict mapping player -> TrueSkill rating object
# # simulate all matches being played, in order
# for paper in papers:
#     play_match(paper, trueskill_rankings)
# rankings_ts = [trueskill_rankings[author].mu for author in authors] # deterministic TrueSkill ratings list

# Uniform Weight Ranking (w/o Label) ----------------------------------------------------------------------------------------------
print("Uniform Weight Ranking ---------------")
pi_list = papers
universe = np.array(list(authors))
numhedges = len(pi_list)
numnodes = len(universe)
# first create these matrices
# R = |E| x |V|, R(e, v) = lambda_e(v)
# W = |V| x |E|, W(v, e) = w(e) 1(v in e)
m = len(pi_list) # number of hyperedges
n = len(universe) # number of items to be ranked 
R_rows, R_cols, R_data = [], [], []
W_rows, W_cols, W_data = [], [], []

for i in range(len(pi_list)):
    pi, scores = pi_list[i]
    if len(pi) > 1:   
        for j in range(len(pi)):
            v = pi[j]
            # v = np.where(universe == v)[0][0] #equivalent to universe.index(v) but for np arrays
            R_rows.append(i)
            R_cols.append(v)
            R_data.append(1.0)
            W_rows.append(v)
            W_cols.append(i)
            W_data.append(1.0)
R = sparse.csr_matrix((R_data, (R_rows, R_cols)), shape=(numhedges, numnodes))
R = normalize(R, norm='l1', axis=1)
W = sparse.csr_matrix((W_data, (W_rows, W_cols)), shape=(numnodes, numhedges))
Wnorm = normalize(W, norm='l1', axis=1)
# create prob trans matrices
P = Wnorm * R
P = P.T
# create rankings
r=0.40
rankings_unif = compute_pr(P, r, n, eps=1e-8).flatten()
del R, W, P

# Predicted Weight Ranking ----------------------------------------------------------------------------------------------
pi_list = papers
universe = np.array(list(authors))
numhedges = len(pi_list)
numnodes = len(universe)

outputdir = "../train_results/AMiner/"
predict_scores = []
with open(outputdir + "prediction.txt", "r") as f:
    for line in f.readlines():
        tmp = line.rstrip().split("\t")
        scores = []
        for i in tmp:
            if int(i) == 0:
                scores.append(2.0)
            elif int(i) == 1:
                scores.append(1.0)
            elif int(i) == 2:
                scores.append(2.0)
        predict_scores.append(scores)
    
m = len(pi_list) # number of hyperedges
n = len(universe) # number of items to be ranked 
R_rows, R_cols, R_data = [], [], []
W_rows, W_cols, W_data = [], [], []

for i in range(len(pi_list)):
    pi, _ = pi_list[i]
    pred = predict_scores[i]
    if len(pi) > 1:   
        for j in range(len(pi)):
            v = pi[j]
            # v = np.where(universe == v)[0][0] #equivalent to universe.index(v) but for np arrays
            R_rows.append(i)
            R_cols.append(v)
            R_data.append(pred[j])
            
            W_rows.append(v)
            W_cols.append(i)
            W_data.append(hindex2cite[i])

R = sparse.csr_matrix((R_data, (R_rows, R_cols)), shape=(numhedges, numnodes))
R = normalize(R, norm='l1', axis=1)
W = sparse.csr_matrix((W_data, (W_rows, W_cols)), shape=(numnodes, numhedges))
Wnorm = normalize(W, norm='l1', axis=1)
# create prob trans matrices
P = Wnorm * R
P = P.T
# create rankings
r=0.40
rankings_bw = compute_pr(P, r, n, eps=1e-8).flatten()
del P

# Evaluation ----------------------------------------------------------------------------------------------------------------------
from scipy.stats import skew, kendalltau
true_rankings = [vindex2rank[v] for v in range(numnodes)]

#pearson product-moment correlation
results_hg = np.corrcoef(np.array([true_rankings, rankings_hg]))[0][1]
results_gh = np.corrcoef(np.array([true_rankings, rankings_gh]))[0][1]
results_mc3 = np.corrcoef(np.array([true_rankings, rankings_mc3]))[0][1]
results_unif = np.corrcoef(np.array([true_rankings, rankings_unif]))[0][1]
results_bw = np.corrcoef(np.array([true_rankings, rankings_bw]))[0][1]
print('Hypergraph w/ GroundTruth accuracy: {}'.format(results_hg))
print('Clique Graph accuracy: {}'.format(results_gh))
print('Dwork MC3 accuracy: {}'.format(results_mc3))
print('Hypergraph w/o Labels accuracy: {}'.format(results_unif))
print('Hypergraph w/ WHATsNet accuracy: {}'.format(results_bw))
print()

results_hg = scipy.stats.spearmanr(true_rankings, rankings_hg)
results_gh = scipy.stats.spearmanr(true_rankings, rankings_gh)
results_mc3 = scipy.stats.spearmanr(true_rankings, rankings_mc3)
results_unif = scipy.stats.spearmanr(true_rankings, rankings_unif)
results_bw = scipy.stats.spearmanr(true_rankings, rankings_bw)

print('Hypergraph w/ GroundTruth ', results_hg)
print('Clique Graph accuracy ', results_gh)
print('Dwork MC3 accuracy ', results_mc3)
print('Hypergraph w/o Labels accuracy ', results_unif)
print('Hypergraph w/ WHATsNet accuracy ', results_bw)
print()

# Evaluation ----------------------------------------------------------------------------------------------------------------------
print("Evaluation ---------------")
def eval_game_h2h(game_players, game_scores, all_players, ranks):
    players_ranked_prev = [player for player in game_players if player in all_players]
    if len(players_ranked_prev) == 2:
        # get scores for players previously ranked
        scores_prev = [game_scores[game_players.index(player)] for player in players_ranked_prev]
        ranks_prev = [ranks[all_players.index(player)] for player in players_ranked_prev]

        # make sure there isn't a tie
        if scores_prev[0] != scores_prev[1]:
            can_eval = True
            
            # check if ranked correctly
            if sum(np.argsort(scores_prev) == np.argsort(ranks_prev)) == 2:
                res = True
            else:
                res = False
        else:
            can_eval = False
            res = False
    else:
        can_eval = False
        res = False
    return (can_eval, int(res))

results_hg=0
results_gh=0
results_unif=0
results_bw=0

total = 0

for ai in tqdm.trange(len(authors)):
    for aj in range(ai+1, len(authors)):
        author_i = authors[ai]
        author_j = authors[aj]
        ans_rank_diff = vindex2rank[author_i] - vindex2rank[author_j]
        if ans_rank_diff == 0:
            continue
        # if abs(cur_ranks[0] - cur_ranks[1]) > 10:
        #     continue
        
        total  += 1
        # evaluate game
        pred_hg = rankings_hg[author_i] - rankings_hg[author_j]
        if ans_rank_diff * pred_hg > 0:
            results_hg += 1        
        pred_gh = rankings_gh[author_i] - rankings_gh[author_j]
        if ans_rank_diff * pred_gh > 0:
            results_gh += 1
        pred_unif = rankings_unif[author_i] - rankings_unif[author_j]
        if ans_rank_diff * pred_unif > 0:
            results_unif += 1
        pred_bw = rankings_bw[author_i] - rankings_bw[author_j]
        if ans_rank_diff * pred_bw > 0:
            results_bw += 1

print('Hypergraph w/ GroundTruth accuracy: {}'.format(results_hg / total))
print('Clique Graph accuracy: {}'.format(results_gh / total))
print('Hypergraph w/o Labels accuracy: {}'.format(results_unif / total))
print('Hypergraph w/ WHATsNet accuracy: {}'.format(results_bw / total))