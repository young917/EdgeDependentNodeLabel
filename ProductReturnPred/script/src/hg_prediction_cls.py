import pickle
import numpy as np
import pandas as pd
# import matplotlib.pylab as plt
from multiprocessing import Pool
from sklearn import metrics
# from scoop import futures

from itertools import compress
from src.hyper_graph import HyperGraph
from src.nibble import nibble
from src.functions import f_point_5, argmax


def _HypergraphPredictor__eval_i(args):
    g_i = args[0]
    b = args[1]
    phi = args[2]
    wgt = args[3]
    last_idx = len(g_i.vertex_name)-1
    res = nibble(g_i, last_idx, b, phi)

    res = res[res != last_idx]
    if(len(res) != 0):
        res_label = g_i.get_labels(res)
        res_h = g_i.h_mat[res,:].sign()
        res_r = g_i.r_mat[res,:].sign()
        pred = (res_label.sum()*0.5 +
                (g_i.r_mat[res,:].dot(g_i.h_mat[last_idx, :].T) > 0).sum()*0.5)/res.shape[0]
        # TODO: this is to try a different prediction method
        pred_prod = (res_h.T.dot(res_label))/(res_h.sum(0).A1)
        with np.errstate(divide='ignore',invalid='ignore'):
            pred_prod = (res_r.sum(0).A1)/(res_h.sum(0).A1)
        idx = np.isnan(pred_prod)
        pred_prod[idx] = g_i.wgt[idx]
    else:
        pred = np.nan
        pred_prod = g_i.h_mat[:last_idx, :].sum(1)

    pred = pred * wgt
    lbl = g_i.get_label(last_idx)


    prd_order = (g_i.h_mat[last_idx,:] > 0).todense().A1
    lbl_prod = np.sign(g_i.r_mat[last_idx,:].todense().A1)
    lbl_prod = lbl_prod[prd_order]

    num_ret = sum(lbl_prod)
    res_h1 = (g_i.h_mat[last_idx, :] > 1).todense().A1*1.0+1

    pred_prod = pred_prod * res_h1
    pred_prod = pred_prod[prd_order]
    # ind = np.argpartition(pred_prod, -num_ret)[-num_ret:]
    # pred_prod = pred_prod * 0
    # pred_prod[ind] = 1

    return pred, lbl, pred_prod, lbl_prod


class HypergraphPredictor:
    """
    Our model
    """
    def __init__(self, max_num_test, parallel, n_cpu, chunk_size):
        """

        :param parallel: a string among the value of ("Single", "Multi", "Scoop")
        :param max_num_test:
        :param n_cpu:
        :param chunk_size:
        """
        self.max_num_test = max_num_test
        self.parallel = parallel
        self.n_cpu = n_cpu
        self.chunk_size = chunk_size
        self.ratio = 1
        self.bsk_label_train = None
        self.g = None

    def fit(self, h_train, bsk_label_train, order_no_train,
            khk_ean_train, return_rate_train, r_train,
            ratio = None):
        """

        :param h_train:
        :param bsk_label_train:
        :param multi_item_bsk:
        :param ratio:
        :return:
        """
        multi_item_bsk = (h_train >1).nonzero()[0]
        multi_idx = np.zeros(h_train.shape[0], dtype=bool)
        multi_idx[multi_item_bsk] = True
        if ratio is None:
            ratio = bsk_label_train[multi_idx].sum()[0]/multi_idx.sum() / \
                (bsk_label_train[np.logical_not(multi_idx)].sum()[0]/np.logical_not(multi_idx).sum())
        self.ratio = ratio
        self.bsk_label_train = bsk_label_train.values/ratio

        self.g = HyperGraph(order_no_train, khk_ean_train,
                       return_rate_train['RET_Items'].values,
                       h_train, bsk_label_train['RET_Items'].values,
                       r_train)


    def predict(self, h_test, bsk_label_test, order_no_test, r_test, b, phi):
        """

        :param h_test:
        :param test_bsk_name:
        :return: since order are not preserved, label are returned together with prediction
        """
        # this is assume we are in python 3!!
        multi_item_bsk = (h_test >1).nonzero()[0]
        multi_idx = np.zeros(h_test.shape[0], dtype=bool)
        multi_idx[multi_item_bsk] = True

        wgt = np.ones((h_test.shape[0]))
        wgt[multi_idx] = self.ratio

        if self.parallel == "Multi":
            pool = Pool(processes=self.n_cpu)
            pred_rst = \
                pool.imap_unordered(__eval_i,
                                    ((self.g.insert(order_no_test[i], h_test[i, :], bsk_label_test.iloc[i, 0],
                                               r_test[i,:], False),
                                      b, phi, wgt[i])
                                     for i in np.arange(0, min(self.max_num_test, len(order_no_test)))),
                                    chunksize=self.chunk_size)
            pred_rst = list(pred_rst)
            pool.close()
        else:
            all_test = map(lambda i: (self.g.insert(order_no_test[i], h_test[i, :],
                                               bsk_label_test.iloc[i, 0],
                                               r_test[i, :], False),
                                      b, phi, wgt[i]),
                           np.arange(0, min(self.max_num_test, len(order_no_test))))
            if self.parallel == "Single":
                pred_rst = list()
                for g_i in all_test:
                    pred_rst.append(__eval_i(g_i))
            elif self.parallel == "Scoop":
                pred_rst = list(futures.map(__eval_i, all_test))
            else:
                raise "Parallel Methods not supported!"

        pred_rst = pd.DataFrame(pred_rst, columns=['pred_prob', 'obs', 'pred_prob_prod', 'obs_prod'])
        return pred_rst

    def pred_test_based_on_valid(self, h_validate, bsk_label_valid, order_no_validate, r_validate,
                                 h_test, bsk_label_test, order_no_test, r_test,
                                 bs, phis):
        """

        :param h_validate:
        :param bsk_label_valid:
        :param h_test:
        :return: optimal prec, recall, f_0.5, auc, fpr, tpr
        """
        n = len(bs)
        m = len(phis)
        best_f = np.zeros((n, m))
        best_auc = np.zeros((n, m))
        thr_opt = np.zeros((n, m))

        for i in range(n):
            b = bs[i]
            for j in range(m):
                phi = phis[j]
                pred_rst = self.predict(h_validate, bsk_label_valid, order_no_validate, r_validate, b, phi)

                fpr, tpr, _ = metrics.roc_curve(pred_rst['obs'], pred_rst['pred_prob'], pos_label=True)
                auc = metrics.auc(fpr, tpr)

                prec, rec, thr = \
                    metrics.precision_recall_curve(pred_rst['obs'], pred_rst['pred_prob'], pos_label=True)
                f1 = f_point_5(prec, rec)
                f1[np.isnan(f1)] = 0
                best_f[i, j] = np.max(f1)
                best_auc[i, j] = auc
                thr_opt[i, j] = thr[np.argmax(f1)]

        idx_k = argmax(best_f, best_auc)
        pred_rst = self.predict(h_test, bsk_label_test, order_no_test, r_test, bs[idx_k[0]], phis[idx_k[1]])

        prec, rec, thr = metrics.precision_recall_curve(pred_rst['obs'], pred_rst['pred_prob'], pos_label=True)
        f = f_point_5(prec, rec)
        idx_f = np.where(thr <= thr_opt[idx_k])[0][-1]

        fpr, tpr, _ = metrics.roc_curve(pred_rst['obs'], pred_rst['pred_prob'], pos_label=True)
        auc = metrics.auc(fpr, tpr)

        return prec[idx_f], rec[idx_f], f[idx_f], auc, fpr, tpr, thr[idx_f], bs[idx_k[0]], phis[idx_k[1]]

    def pred_test_based_on_valid_prod(self, h_validate, bsk_label_valid, order_no_validate, r_validate,
                                 h_test, bsk_label_test, order_no_test, r_test,
                                 bs, phis):
        """

        :param h_validate:
        :param bsk_label_valid:
        :param h_test:
        :return: optimal prec, recall, f_0.5, auc, fpr, tpr
        """
        n = len(bs)
        m = len(phis)
        best_f = np.zeros((n, m))
        best_auc = np.zeros((n, m))
        thr_opt = np.zeros((n, m))

        for i in range(n):
            b = bs[i]
            for j in range(m):
                phi = phis[j]
                print("b=%f, phi=%f" % (b, phi))
                pred_rst = self.predict(h_validate, bsk_label_valid, order_no_validate, r_validate, b, phi)
                pred_rst_prod = pred_rst[['obs_prod', 'pred_prob_prod']]

                obs_prod = [item for l in pred_rst_prod['obs_prod'] for item in l]
                pred_prod = [item for l in pred_rst_prod['pred_prob_prod'] for item in l]

                fpr, tpr, _ = metrics.roc_curve(obs_prod, pred_prod, pos_label=True)
                auc = metrics.auc(fpr, tpr)

                prec, rec, thr = \
                    metrics.precision_recall_curve(obs_prod, pred_prod, pos_label=True)
                f1 = f_point_5(prec, rec)
                f1[np.isnan(f1)] = 0
                best_f[i, j] = np.max(f1)
                best_auc[i, j] = auc
                thr_opt[i, j] = thr[np.argmax(f1)]

        idx_k = argmax(best_f, best_auc)
        pred_rst = self.predict(h_test, bsk_label_test, order_no_test, r_test, bs[idx_k[0]], phis[idx_k[1]])
        obs_prod = [item for l in pred_rst['obs_prod'] for item in l]
        pred_prod = [item for l in pred_rst['pred_prob_prod'] for item in l]
        prec, rec, thr = metrics.precision_recall_curve(obs_prod, pred_prod, pos_label=True)
        f = f_point_5(prec, rec)
        idx_f = np.where(thr <= thr_opt[idx_k])[0]
        if len(idx_f) > 0:
            idx_f = idx_f[-1]
        else:
            idx_f = 0
        fpr, tpr, _ = metrics.roc_curve(obs_prod, pred_prod, pos_label=True)
        auc = metrics.auc(fpr, tpr)

        return prec[idx_f], rec[idx_f], f[idx_f], auc, fpr, tpr, thr[idx_f], bs[idx_k[0]], phis[idx_k[1]]

if __name__ == "__main__":
    import matplotlib.pylab as plt
    with open("../data/order_no_train.pkl", 'rb') as f:
        order_no_train = pickle.load(f)
    with open("../data/khk_ean_train.pkl", 'rb') as f:
        khk_ean_train = pickle.load(f)
    bsk_label_train = pd.read_pickle("../data/bsk_label_train.pkl")
    return_rate_train = pd.read_pickle("../data/return_rate_train.pkl")

    with open("../data/h_train.pkl", 'rb') as f:
        h_train = pickle.load(f)
    with open("../data/r_train.pkl", 'rb') as f:
        r_train = pickle.load(f)

    with open("../data/h_validate.pkl", 'rb') as f:
        h_validate = pickle.load(f)
    with open("../data/r_validate.pkl", 'rb') as f:
        r_validate = pickle.load(f)
    with open("../data/order_no_validate.pkl", 'rb') as f:
        order_no_validate = pickle.load(f)
    bsk_label_validate = pd.read_pickle("../data/bsk_label_validate.pkl")


    with open("../data/h_test.pkl", 'rb') as f:
        h_test = pickle.load(f)
    with open("../data/r_test.pkl", 'rb') as f:
        r_test = pickle.load(f)
    with open("../data/order_no_test.pkl", 'rb') as f:
        order_no_test = pickle.load(f)
    bsk_label_test = pd.read_pickle("../data/bsk_label_test.pkl")

    # Construct graph ====================================
    p = HypergraphPredictor(max_num_test=10, parallel="Single", n_cpu=7, chunk_size=1)
    p.fit(h_train, bsk_label_train, order_no_train, khk_ean_train,
          return_rate_train, r_train, ratio=None)

    h_validate = h_validate[bsk_label_validate['RET_Items'].values,:]
    order_no_validate = list(compress(order_no_validate, bsk_label_validate['RET_Items'].values))
    r_validate = r_validate[bsk_label_validate['RET_Items'].values, :]
    bsk_label_validate = bsk_label_validate[bsk_label_validate['RET_Items'].values]

    h_test = h_test[bsk_label_test['RET_Items'].values,:]
    order_no_test = list(compress(order_no_test, bsk_label_test['RET_Items'].values))
    r_test = r_test[bsk_label_test['RET_Items'].values, :]
    bsk_label_test = bsk_label_test[bsk_label_test['RET_Items'].values]

    prec, rec, f, auc, fpr, tpr, thr, b, phi = \
        p.pred_test_based_on_valid_prod(h_validate, bsk_label_validate, order_no_validate, r_validate,
                                   h_test, bsk_label_test, order_no_test, r_test,
                                   [6, 7, 8, 9, 10],
                                   [0.4, 0.6, 0.8])

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'b--')

    print("auc, prec, rec, f1, thr, b, phi:")
    print("%f, %f, %f, %f, %f, %f, %f" % (auc, prec, rec, f, thr, b, phi))
