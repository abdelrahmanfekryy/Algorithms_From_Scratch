import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, minPoints=5):
        self.eps = eps
        self.minPoints = minPoints

    def dbscan_inner(self,is_core,neighborhoods,labels):
        label_num = 0
        stack = []
        for i in range(labels.shape[0]):
            if labels[i] != -1 or not is_core[i]:
                continue
            while True:
                if labels[i] == -1:
                    labels[i] = label_num
                    if is_core[i]:
                        for idx in np.where(neighborhoods[i])[0]:
                            if labels[idx] == -1:
                                stack.append(idx)

                if len(stack) == 0:
                    break
                i = stack.pop()
            label_num += 1

    def fit(self, X):
        self.neighbors = np.linalg.norm(X - np.expand_dims(X, axis=1),axis=-1) < self.eps
        n_neighbors = np.sum(self.neighbors,axis=1)
        labels = np.full(X.shape[0], -1, dtype=int)
        core_idxs = n_neighbors >= self.minPoints
        self.dbscan_inner(core_idxs,self.neighbors, labels)
        self.labels_ = labels
        noise_idxs = self.labels_ == -1
        self.reachability = np.ones(X.shape[0],dtype=int)
        self.reachability[core_idxs] = 0
        self.reachability[noise_idxs] = -1