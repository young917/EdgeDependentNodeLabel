#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:41:42 2017

@author: bob
"""

import bisect
# import matplotlib.pylab as plt
import pandas as pd
from scipy import sparse
# %% define Nibble ===========================================================
import numpy as np

from src.hyper_graph import HyperGraph

def truncat(p, threshold):
    """
        define the truncation function in step 2
    """
    for idx in p.nonzero()[0]:
        if (p[idx, 0] <= threshold[idx]):
            p[idx, 0] = 0
    return p

def approx_sj2(g, q):
    idx_j = q.nonzero()
    q_nnz = q[idx_j]
    # TODO: We added divided by Pi here
    pi = g.stationary_dist[idx_j[0]]
    if isinstance(q_nnz, sparse.csr.csr_matrix):
        q_nnz = (q/pi).todense()

    q_nnz = np.squeeze(np.asarray(q_nnz))
    idx = np.argsort(q_nnz)[::-1]
    return idx_j[0][idx]


def nibble(g, v, b, phi):
    """
       Function Nibble is the implementation of the adaptive Nibble algorithm
       g: the graph (dictionary)
       v: the advertiser node (string)
       b: the size of cluster b > k (integer)
       phi: the upper bound on the conductance (0 < phi < 1)
       unode: the user node lists
    """

    # constant parameters
    c1 = 200
    c3 = 1800
    c4 = 140

    degree = g.degrees

    mu_V = sum(degree)
    stat_dist = g.stationary_dist
     # step 1

    l = int(np.ceil(np.log2(mu_V / 2.)))
    t1 = int(np.ceil(2. / phi ** 2 *
                     np.log(c1 * (l + 2) * np.sqrt(mu_V / 2.))))
    tLast = (l + 1) * t1
    epsilon = 1. / (c3 * (l + 2) * tLast * 2 ** b)

    # %%=======================================================================
    # step 2
    # assuming that all the vertices are indexed from 0 to Len
    chi_v = sparse.coo_matrix(([1.], ([v], [0])), (g.len(), 1))
    q = chi_v.tocsr()

    r = truncat(q, stat_dist * epsilon)

    # %%=======================================================================
    # step 3

    M = g.lazy_transition_mat.T.tocsr()

    if 2 ** b >= 5. / 6 * mu_V:
        print("""b (%f) is too large: 2**b > 5./6*mu_V. It should be less than %f""" %
              (b, np.log(5. / 6 * mu_V) / np.log(2)))
        return []

    # %%=======================================================================
    #    numOfLastCust = 0
    timeOfStayStill = 0
    numOfLastNode = 0

    for t in range(tLast):
        #if t % 1000 == 0:

        # logging.debug('running inside Nibble for t loop v is  %s  t is %s ',
        #               v,t)
        q = M.dot(r)
        r = truncat(q, stat_dist * epsilon)

        # start = time.time()
        # get all the sorted none zero concentration nodes index
        # TODO: r might be better
        idx = approx_sj2(g, q)
        #
        numOfNode = len(idx)

        if numOfNode == numOfLastNode:
            timeOfStayStill += 1
        else:
            timeOfStayStill = 0
            numOfLastNode = numOfNode

        if timeOfStayStill > 10:  # no change after 10 times of iteration
            return idx

  
        for j in range(1, len(idx)):  # allJ[0]:
            # start_j =time.time()
            idxJ = idx[:j]
            # lambdaJ = sum(out_dgreeU[idxJ] + in_degreeU[idxJ])/2.
            lambdaJ = sum(degree[idxJ])

            # condition C2 and C3
            if not (lambdaJ <= 5. / 6 * mu_V and 2 ** b <= lambdaJ):
                continue

            # condition C1

            Phi_j = g.boundary_vol(idxJ)

            if Phi_j > phi:
                continue
            # Condition C4
            qIdxJ = np.squeeze(np.asarray(q[idxJ].todense()))
            #lambdas = np.cumsum(out_dgreeU[idxJ] + in_degreeU[idxJ])/2.
            lambdas = np.cumsum(degree[idxJ])
            # the jj statisfies lambda_jj(q_t) <= 2**b <= lambda_jj+1(q_t)
            idxJJ = bisect.bisect(lambdas, 2 ** b)
            if idxJJ == 0:
                print("b is probably too small")
                return np.array([])

            # calculate Ix for each j using ranked q
            if idxJJ == len(qIdxJ):
                continue

            IxJ = qIdxJ[idxJJ] / stat_dist[idxJ][idxJJ]

            if IxJ >= 1. / (c4 * (l + 2) * 2 ** b):
                sJ = idxJ
                return sJ
            else:
                break  # this can be done as IxJ depends on k rather than j
    return np.array([])


if __name__ == "__main__":
    import pickle
    with open("../data/order_no.pkl", 'rb') as f:
        order_no = pickle.load(f)
    with open("../data/style_color.pkl", 'rb') as f:
        khk_ean = pickle.load(f)

    with open("../data/h_mat.pkl", 'rb') as f:
        h = pickle.load(f)
    with open("../data/r_mat.pkl", 'rb') as f:
        r = pickle.load(f)

    #return_rate = pd.read_pickle("../data/return_rate.pkl")
    #bsk_label = pd.read_pickle("../data/bsk_return_label.pkl")
    bsk_label = pd.DataFrame(r.sum(axis=1)>0, index=order_no, columns=['RET_Items'])
    return_rate = pd.DataFrame(((r.sum(axis=0)+1)/(h.sum(axis=0)+1)).T, index=khk_ean, columns=['RET_Items'])


    return_rate = return_rate.loc[khk_ean, :]
    bsk_label = bsk_label.loc[order_no, :]
    g = HyperGraph(order_no, khk_ean, return_rate['RET_Items'].values,
                   h, bsk_label['RET_Items'].values, r)

    b = 4
    phi = 0.5
    res = nibble(g, 1, b, phi)
    print(res)
