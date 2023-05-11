import csv
from datetime import datetime
import time
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import weightedtau, kendalltau
from scipy.stats import norm
from scipy.linalg import null_space
import pandas as pd

from pprint import pprint
from copy import deepcopy

import trueskill
import os
import argparse

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--outputname', type=str, default="our_0")
args = parser.parse_args()



dt_str = '%d %B %Y %H:%M:%S'

dt_lim = datetime.strptime('06 August 2004 18:13:50', dt_str) #first game in HeadToHead

players = set()

# each entry is a tuple ([list of players], [list of scores])
matches = []

# needed for iteration
cur_game = -1
cur_players = []
cur_scores = []

# also, filter out all games where every player has no score!
with open('FreeForAll.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        date = datetime.strptime(row[0], dt_str)
        score = int(row[6])
        if date < dt_lim:
            game = int(row[1])
            player = row[4]
            score = int(row[6])
            
            # next, decide if this row is from the same match
            # as the last row, or a different match
            if game == cur_game:
                cur_players.append(player)
                cur_scores.append(score)
            else:
                if cur_game > 0 and np.sum(np.abs(cur_scores)):
                    # append cur_players, cur_scores to matches
                    matches.append((cur_players, cur_scores))
                    # add cur_players to players
                    players.update(cur_players)

                # reset cur_game, cur_players, cur_scores
                cur_game = game
                cur_players = [player]
                cur_scores = [score]
        else:
            break

players=list(players) # list of players

print(len(matches))
print(len(players))

avg_size = 0
nnz = 0
for i in range(len(matches)):
    pi, score = matches[i]
    nnz += len(pi)
    avg_size += len(pi)
print(nnz / (len(matches) * len(players)))
print(avg_size / len(matches))


##################################################
# COMPUTE PAGERANK
##################################################

# given probability transition matrix P
# where P_{v,w} = Prob(w -> v)
# find pagerank scores with restart probability r
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

pi_list = matches
universe = np.array(list(players))
# first create these matrices
# R = |E| x |V|, R(e, v) = lambda_e(v)
# W = |V| x |E|, W(v, e) = w(e) 1(v in e)
    
##################################################
# PREDICT
##################################################
outputdir = "../train_results/halo/"# path for saved model
pi_list = matches
universe = np.array(list(players))
# first create these matrices
# R = |E| x |V|, R(e, v) = lambda_e(v)
# W = |V| x |E|, W(v, e) = w(e) 1(v in e)
m = len(pi_list) # number of hyperedges
n = len(universe) # number of items to be ranked 
R = np.zeros([m, n])
W = np.zeros([n, m])

predict_scores = []
with open(outputdir + "prediction_{}.txt".format(args.outputname), "r") as f:
    for line in f.readlines():
        scores = [float(s) for s in line.rstrip().split("\t")]
        predict_scores.append(scores)
for i in range(len(pi_list)):
    pi = pi_list[i][0]
    scores = predict_scores[i]
    assert len(pi) == len(scores)
    
    if len(pi) > 1:   
        binned_scores = []
        for j in range(len(pi)):
            v = pi[j]
            v = np.where(universe == v)[0][0]
            sc = scores[j]
            R[i, v] = sc 
            W[v,i] = 1.0
            binned_scores.append(np.log(sc))
        W[:, i] = (np.std(binned_scores) + 1.0) * W[:, i]
        R[i, :] = R[i,:] / sum(R[i,:])

# first, normalize W
Wnorm=W/W.sum(axis=1)[:,None]
Ws = sparse.csr_matrix(Wnorm)
Rs = sparse.csr_matrix(R)
# create prob trans matrices
P = np.transpose(Ws.dot(Rs))
# create rankings
r=0.40
rankings_bw = compute_pr(P, r, n, eps=1e-8).flatten()


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

cur_game = -1
cur_players = []
cur_scores = []
results_bw=[]

with open('HeadToHead.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        game = int(row[1])
        player = row[4]
        score = int(row[6])

        # next, decide if this row is from the same match
        # as the last row, or a different match
        if game == cur_game:
            cur_players.append(player)
            cur_scores.append(score)
        else:
            if cur_game > 0 and np.sum(np.abs(cur_scores)) > 0:
                can_eval, bw_match_res = eval_game_h2h(cur_players, cur_scores, players, rankings_bw)
                
                if can_eval:
                    results_bw.append(bw_match_res)

            # reset cur_game, cur_players, cur_scores
            cur_game = game
            cur_players = [player]
            cur_scores = [score]

score = sum(results_bw) * 1.0 / len(results_bw)
print('Hypergraph w/ WHATsNet accuracy: {}'.format(score))

if os.path.isdir("../train_results/halo/res/") is False:
    os.makedirs("../train_results/halo/res/")
outputname = args.outputname.split("_")[0]
seed = args.outputname.split("_")[1]
with open("../train_results/halo/res/{}.txt".format(outputname), "+a") as f:
    f.write(",".join([seed, str(score)]) + "\n")