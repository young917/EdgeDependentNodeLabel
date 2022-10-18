import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.sparse import diags
# import matplotlib.pylab as plt

#pylint: disable-msg=too-many-instance-attributes
class HyperGraph(object):
    """
    A labelled hypergraph ready for nibble
    """
#pylint: disable-msg=too-many-arguments
    def __init__(self, vertex_name, edge_name, wgt, h_mat, labels, r_mat, mutli_item_label=None):
        """
        The constructor of hyper_graph
        :param vertex_name: name of the vertex
        :param edge_name: name of the edges
        :param wgt: weight of each edge
        :param h_mat: h matrix
        :param labels: label of the vertex

        Please note: all the order are assumed to be aligned
        """
        self.vertex_name = vertex_name
        self.edge_name = edge_name
        self.wgt = wgt
        self.h_mat = h_mat
        self.labels = labels
        self.r_mat = r_mat
        self.mult_item_label = mutli_item_label

        self.degrees = self.h_mat.dot(self.wgt)
        self.edge_degrees = self.h_mat.sum(0).A1
        self.vertex_indexes = dict(zip(vertex_name, range(len(vertex_name))))
        self.edge_index = dict(zip(edge_name, range(len(edge_name))))

        #self.stationary_dist = self.degrees/sum(self.degrees)
        lazy_transition_mat = diags(1./self.degrees).\
            dot(h_mat > 0).dot(diags(wgt)).\
            dot(diags(1./self.edge_degrees)).\
            dot(h_mat.T > 0)
        # TODO: NOT Lazy form
        # self.lazy_transition_mat = lazy_transition_mat
        # self.stationary_dist = self.degrees/self.degrees.sum()
        # self.adjacency_matrix = h_mat.dot(sparse.diags(wgt)).\
        #     dot(h_mat.transpose()) - sparse.diags(self.degrees)
        tot = lazy_transition_mat.sum(1).A1
        self.lazy_transition_mat = diags(1./tot).dot(lazy_transition_mat)
        self.stationary_dist = self.__station_dist()


    def vertex_degree(self, vertex_idx):

        if type(vertex_idx) is list:
            vertex_idx = np.array(vertex_idx)
        return self.degrees[vertex_idx]

    def degree_all_vertices(self):
        """degree of all nodes"""
        return self.degrees

    def len(self):
        """total number of nodes in the graph"""
        return len(self.vertex_name)

    def vertices(self):
        """return all nodes in the graph"""
        # return self.vertex_indexes.values()
        return np.arange(0, len(self.vertex_name))

    def get_label(self, vertex):
        """return the label of the Vertex"""
        return self.labels[vertex]


    def get_labels(self, vertices):
        """return the label of a Vertex vector"""
        if isinstance(vertices, list):
            vertices = np.array(vertices)
        return self.labels[vertices]

    def get_multi_item_labels(self, vertices):
        """return the label of a Vertex vector"""
        if self.mult_item_label is None:
            return None

        if isinstance(vertices, list):
            vertices = np.array(vertices)
        return self.mult_item_label[vertices]

    def edge_degree(self, edge_idx):
        """return the degree of edges"""
        if isinstance(edge_idx, list):
            edge_idx = np.array(edge_idx)
        return self.edge_degrees[edge_idx]

    def edge_all_vertices(self):
        """return the degree of all vertices"""
        return self.edge_degrees 

    def boundary_vol(self, idx):
        """
        Calculate the boundary volume of a cluster defined by idx
        :param idx: the index of vertices belongs to the cluster
        :return: the volume of the boundary
        """
        idx_tot = range(self.len())
        idx_other = np.setdiff1d(idx_tot, idx)
        cut_edge = (self.h_mat[idx, :].sum(0).A1 * self.h_mat[idx_other, :].sum(0).A1
                    * self.wgt / self.edge_degrees).sum()
        return cut_edge*(1./self.degrees[idx].sum() + 1/self.degrees[idx_other].sum())

    def __station_dist(self):
        stat_dist = self.degrees/self.degrees.sum()
        for i in range(100):
            old_stat = stat_dist
            stat_dist = self.lazy_transition_mat.T.dot(stat_dist)
            if (abs(stat_dist-stat_dist)).max() < 1e-5:
                return stat_dist

        return stat_dist


    def insert(self, new_vertex_name, new_h_vec, new_label, new_r_vec, is_multi_item = None, inplace=False):
        """
        Insert a new_vertex to the graph
        :param new_vertex_name: a int/string to represent the new vertex
        :param new_h_vec: the row will be append to the h_mat
        :param new_label: the label of the new vertex
        :param inplace: if the graph is directly changed or after a deep copy
        :return: If inplace then there is no return value, otherwise a deep copy will be returned
        """
        if inplace:
            raise NotImplementedError("To be implemented")
        else:
            vertex_name = self.vertex_name + [new_vertex_name]
            h_mat = sparse.vstack((self.h_mat, new_h_vec))
            labels = np.hstack((self.labels, new_label))
            r_mat = sparse.vstack((self.r_mat, new_r_vec))
            if (self.mult_item_label is not None) and (is_multi_item is not None):
                multi_item_label = np.hstack((self.mult_item_label, is_multi_item))
            else:
                multi_item_label = None
            return HyperGraph(vertex_name, list(self.edge_name), np.copy(self.wgt), h_mat, labels, r_mat, multi_item_label)



if __name__ == "__main__":
    import pickle
    import matplotlib.pylab as plt
    with open("../data/order_no.pkl", 'rb') as f:
        order_no = pickle.load(f)
    with open("../data/style_color.pkl", 'rb') as f:
        khk_ean = pickle.load(f)
    with open("../data/h_mat.pkl", 'rb') as f:
        h = pickle.load(f)
    with open("../data/r_mat.pkl", 'rb') as f:
        r = pickle.load(f)
    #bsk_label = pd.read_pickle("../data/bsk_return_label.pkl")
    #return_rate = pd.read_pickle("../data/return_rate.pkl")
    bsk_label = pd.DataFrame(r.sum(axis=1)>0, index=order_no, columns=['RET_Items'])
    return_rate = pd.DataFrame(((r.sum(axis=0)+1)/(h.sum(axis=0)+1)).T, index=khk_ean, columns=['RET_Items'])

    return_rate = return_rate.loc[khk_ean, :]
    bsk_label = bsk_label.loc[order_no, :]
    g = HyperGraph(order_no, khk_ean, return_rate['RET_Items'].values,
                    h, bsk_label['RET_Items'].values, r)

    #h_vec = np.random.randint(0, 1, len(g.edge_name))
    #g1 = g.insert(123456, h_vec, 0, False)

    plt.plot(g.degrees/g.degrees.sum(), 'o')
    plt.plot(g.stationary_dist, 'o')
    plt.show()
