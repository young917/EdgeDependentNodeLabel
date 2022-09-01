import torch
import numpy as np
import pdb
import random

class DataLoader(torch.utils.data.Dataset):

    def __init__(self, hedge2node, node2hedge, hedge2labels, hedge2type, batch_size, n_layers, device, shuffleflag=True, sampling=-1):        
        self.batch_size = batch_size
        self.hedge2node = hedge2node
        self.node2hedge = node2hedge
        self.hedge2labels = hedge2labels
        self.hedge2type = hedge2type
        self.order = list(range(len(self.hedge2node)))
        
        self.idx = 0
        self.n_layers = n_layers
        self.device = device
        if shuffleflag:
            self.shuffle()
        self.type = 0
        self.sampling = sampling
    
    def eval(self, evaltype="valid"):
        if evaltype == "valid":
            self.type = 1
        elif evaltype == "test":
            self.type = 2

    def train(self):
        self.type = 0

    def shuffle(self):
        random.shuffle(self.order)
        
    def __iter__(self):
        self.idx = 0
        return self
    
    def next(self):
        return self.__next__()

    def __next__(self):
        return self.next_batch()

    def next_batch(self):
        is_last = False
        target_hes = []
        _idx = self.idx
        while (len(target_hes) < self.batch_size) and (_idx < len(self.order)):
            hidx = self.order[_idx]
            if self.hedge2type[hidx] == self.type:
                target_hes.append(hidx)
            _idx += 1
            
        if _idx == len(self.order):
            is_last = True
            if self.type == 0:
                _idx = 0
                while (len(target_hes) < self.batch_size):
                    hidx = self.order[_idx]
                    if self.hedge2type[hidx] == self.type:
                        target_hes.append(hidx)
                    _idx += 1
        self.idx = _idx
        # gather neighbor hyperedges by BFS
        check = [False for h in range(len(self.hedge2node))]
        queue_hes = []
        for h in target_hes:
            check[h] = True
            queue_hes.append(h)
        nodeset = set()
        for _ in range(self.n_layers):
            size = len(queue_hes)
            for _ in range(size):
                h = queue_hes.pop(0)
                check[h] = True
                neighbors = []
                for v in self.hedge2node[h]:
                    nodeset.add(v)
                    for nh in self.node2hedge[v]:
                        if check[nh] is False and nh not in neighbors:
                            neighbors.append(nh)
                random.shuffle(neighbors)
                look = min(self.sampling, len(neighbors)) if self.sampling != -1 else len(neighbors)
                for i in range(look):
                    nh = neighbors[i]
                    check[nh] = True
                    queue_hes.append(nh)
        # make batch data
        node2newindex = {}
        nodeorder = [] # i-th node in batch = actually [i] node in dataset
        hedgeorder = [] # i-th hyperedge in batch = actually [i] hyperedge in dataset
        maskhedges = [] # mask hyperedges which are not for calculating loss
        hedge2newnodeindex = []
        hedge2labels = []
        for h in range(len(self.hedge2node)):
            if check[h] is True:
                hedge = []
                labels = []
                hedgeorder.append(h)
                index_shuf = list(range(len(self.hedge2node[h]))) # shuffle nodes in hyperedge
                random.shuffle(index_shuf)
                for i in index_shuf:
                    v = self.hedge2node[h][i]
                    lab = self.hedge2labels[h][i]
                    if v not in node2newindex:
                        node2newindex[v] = len(node2newindex)
                        nodeorder.append(v)
                    newindex = node2newindex[v]
                    hedge.append(newindex)
                    labels.append(lab)
                hedge2labels.append(labels)
                hedge2newnodeindex.append(hedge)
                if h in target_hes:
                    maskhedges.append(1)
                else:
                    maskhedges.append(0)
                assert len(hedge) == len(self.hedge2labels[h])

        maskhedges = torch.LongTensor(maskhedges).to(self.device)
        hedgeorder = torch.LongTensor(hedgeorder).to(self.device)
        nodeorder = torch.LongTensor(nodeorder).to(self.device)
        hedge2newnodeindex = [torch.LongTensor(edge).to(self.device) for edge in hedge2newnodeindex]
        hedge2labels = [torch.LongTensor(labels).to(self.device) for labels in hedge2labels]
        
        for hedge, labels in zip(hedge2newnodeindex, hedge2labels) :
            assert hedge.shape[0] == labels.shape[0]
            
        if is_last :
            self.shuffle()     
            self.idx = 0
            
        return maskhedges, hedgeorder, nodeorder, hedge2newnodeindex, hedge2labels, is_last

class DataLoaderwRank(torch.utils.data.Dataset):

    def __init__(self, hedge2node, hedge2noderank, node2hedge, node2hedgerank, hedge2labels, hedge2type, batch_size, n_layers, device, shuffleflag=True, sampling=-1):        
        self.batch_size = batch_size
        self.hedge2node = hedge2node
        self.hedge2noderank = hedge2noderank
        self.node2hedge = node2hedge
        self.node2hedgerank = node2hedgerank
        self.hedge2labels = hedge2labels
        self.hedge2type = hedge2type
        self.order = list(range(len(self.hedge2node)))
        
        self.idx = 0
        self.n_layers = n_layers
        self.device = device
        if shuffleflag:
            self.shuffle()
        self.type = 0
        self.sampling = sampling
    
    def eval(self, evaltype="valid"):
        if evaltype == "valid":
            self.type = 1
        elif evaltype == "test":
            self.type = 2

    def train(self):
        self.type = 0

    def shuffle(self):
        random.shuffle(self.order)
        
    def __iter__(self):
        self.idx = 0
        return self
    
    def next(self):
        return self.__next__()

    def __next__(self):
        return self.next_batch()

    def next_batch(self):
        is_last = False
        target_hes = []
        _idx = self.idx
        while (len(target_hes) < self.batch_size) and (_idx < len(self.order)):
            hidx = self.order[_idx]
            if self.hedge2type[hidx] == self.type:
                target_hes.append(hidx)
            _idx += 1
            
        if _idx == len(self.order):
            is_last = True
            if self.type == 0:
                _idx = 0
                while (len(target_hes) < self.batch_size):
                    hidx = self.order[_idx]
                    if self.hedge2type[hidx] == self.type:
                        target_hes.append(hidx)
                    _idx += 1
        self.idx = _idx
        # gather neighbor hyperedges by BFS
        check = [False for h in range(len(self.hedge2node))]
        queue_hes = []
        for h in target_hes:
            check[h] = True
            queue_hes.append(h)
        nodeset = set()
        for _ in range(self.n_layers):
            size = len(queue_hes)
            for _ in range(size):
                h = queue_hes.pop(0)
                check[h] = True
                neighbors = []
                for v in self.hedge2node[h]:
                    nodeset.add(v)
                    for nh in self.node2hedge[v]:
                        if check[nh] is False and nh not in neighbors:
                            neighbors.append(nh)
                random.shuffle(neighbors)
                look = min(self.sampling, len(neighbors)) if self.sampling != -1 else len(neighbors)
                for i in range(look):
                    nh = neighbors[i]
                    check[nh] = True
                    queue_hes.append(nh)
        # make batch data
        node2newindex = {}
        nodeorder = []
        hedgeorder = []
        maskhedges = []
        hedge2newnodeindex = []
        hedge2noderankB = []
        node2newhedgeinedex = []
        node2hedgerankB = []
        hedge2labels = []
        for h in range(len(self.hedge2node)):
            if check[h] is True:
                hedge = []
                labels = []
                ranks = []
                hedgeorder.append(h)
                index_shuf = list(range(len(self.hedge2node[h])))
                random.shuffle(index_shuf)
                for i in index_shuf:
                    v = self.hedge2node[h][i]
                    lab = self.hedge2labels[h][i]
                    rank = self.hedge2noderank[h][i]
                    if v not in node2newindex:
                        node2newindex[v] = len(node2newindex)
                        nodeorder.append(v)
                        node2newhedgeinedex.append([])
                        node2hedgerankB.append([])
                    newindex = node2newindex[v]
                    hedge.append(newindex)
                    labels.append(lab)
                    ranks.append(rank)
                    node2newhedgeinedex[newindex].append(len(hedge2newnodeindex))
                    for horder, _h in enumerate(self.node2hedge[v]):
                        if _h == h:
                            node2hedgerankB[newindex].append(self.node2hedgerank[v][horder])
                            break
                hedge2labels.append(labels)
                hedge2newnodeindex.append(hedge)
                hedge2noderankB.append(ranks)
                if h in target_hes:
                    maskhedges.append(1)
                else:
                    maskhedges.append(0)
                    
                assert len(hedge) == len(self.hedge2labels[h])
                assert len(hedge) == len(self.hedge2node[h])
                assert len(hedge) == len(self.hedge2noderank[h])
        
        assert len(hedge2newnodeindex) == len(hedge2noderankB)
        for hidx in range(len(hedge2newnodeindex)):
            assert len(hedge2newnodeindex[hidx]) == len(hedge2noderankB[hidx])
        assert len(nodeorder) == len(node2hedgerankB)
        
        for v in range(len(self.node2hedge)):
            if v in node2newindex:
                length = 0
                for h in self.node2hedge[v]:
                    if check[h] is True:
                        length += 1
                newindex = node2newindex[v]
                assert len(node2hedgerankB[newindex]) == length, str(len(node2hedgerankB[newindex]))
                assert len(node2newhedgeinedex[newindex]) == length
        
        maskhedges = torch.LongTensor(maskhedges).to(self.device)
        hedgeorder = torch.LongTensor(hedgeorder).to(self.device)
        nodeorder = torch.LongTensor(nodeorder).to(self.device)
        hedge2newnodeindex = [torch.LongTensor(edge).to(self.device) for edge in hedge2newnodeindex]
        node2newhedgeinedex = [torch.LongTensor(node).to(self.device) for node in node2newhedgeinedex]
        hedge2labels = [torch.LongTensor(labels).to(self.device) for labels in hedge2labels]
        
        for hedge, labels in zip(hedge2newnodeindex, hedge2labels) :
            assert hedge.shape[0] == labels.shape[0]
            
        if is_last :
            self.shuffle()     
            self.idx = 0
            
        return maskhedges, hedgeorder, nodeorder, hedge2newnodeindex, hedge2noderankB, node2newhedgeinedex, node2hedgerankB, hedge2labels, is_last
