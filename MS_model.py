import numpy as np

class Mean_Shift:
    def __init__(self, min_bin_freq=1,max_iter=300):
        self.max_iter = max_iter
        self.min_bin_freq = min_bin_freq

    def get_bandwidth(self,X,quantile = 0.3):
        n_neighbors = int(X.shape[0] * quantile)
        distance_matrix = np.linalg.norm(X - np.expand_dims(X, axis=1),axis =-1)
        return np.sum(np.sort(distance_matrix,axis = 1)[:,n_neighbors - 1])/distance_matrix.shape[0]

    def init_centers(self,X,bandwidth,min_bin_freq = 1):
        point,idx, freq = np.unique(np.round(X/bandwidth),axis=0,return_counts=True,return_index=True)
        idx = np.argsort(idx)
        bin_seeds = point[idx][freq >= min_bin_freq] * bandwidth
        return bin_seeds

    def shift_centers(self,X,centers,radius, max_iter):
        stop_thresh = 1e-3 * radius
        X_stack = np.repeat(np.expand_dims(X,axis=0),centers.shape[0],axis=0)
        completed_iterations = np.zeros(centers.shape[0])
        centers_history = np.expand_dims(centers.copy(),axis=0)
        for _ in range(max_iter):
            dist = np.linalg.norm(X - np.expand_dims(centers,axis=1),axis=-1)
            centers = np.nanmean(np.where(np.expand_dims(dist < radius,axis=-1),X_stack,np.nan),axis=1)
            diff = np.linalg.norm(centers - centers_history[-1],axis =-1)  > stop_thresh
            centers_history = np.vstack((centers_history,np.expand_dims(centers,axis=0)))
            completed_iterations += diff
            if np.all(~diff):
                break
        n_points_within = np.sum(dist < radius,axis=1)
        return centers,n_points_within,completed_iterations,centers_history

    def filter_centers(self,centers,bandwidth):
        dist = np.linalg.norm(centers - np.expand_dims(centers, axis=1),axis=-1) < bandwidth
        idx = np.ones(centers.shape[0],dtype=bool)
        ids = np.ones(centers.shape[0],dtype=int)
        for i in range(dist.shape[0]):
            if idx[i]:
                idx[dist[i]] = 0
                idx[i] = 1
                ids[dist[i]] = i
        return centers[idx],centers[ids]
 

    def fit(self,X):
        self.bandwidth = self.get_bandwidth(X)
        self.seeds = self.init_centers(X,self.bandwidth,min_bin_freq=1)
        Data_ = self.shift_centers(X,self.seeds,self.bandwidth,self.max_iter)
        self.n_points_within = Data_[1]
        self.centers_history = np.transpose(Data_[3],axes=(1,0,2))
        self.shifted = Data_[0]
        self.max_iter_ = np.max(Data_[2])
        centers,idx = np.unique(Data_[0],axis=0,return_index=True)
        n_points_within = Data_[1][idx]
        sort = np.argsort(n_points_within)[::-1]
        n_points_within = n_points_within[sort]
        self.pre_filtered_centers = centers[sort]
        self.centers,self.filtered_centers = self.filter_centers(self.pre_filtered_centers,self.bandwidth)
        self.labels = np.argmin(np.linalg.norm(X - np.expand_dims(self.centers, axis=1),axis =-1),axis=0)




