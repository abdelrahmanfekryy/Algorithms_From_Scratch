import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

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

    def mean_shift(self,X,centers,radius, max_iter):
        stop_thresh = 1e-3 * radius
        X_stack = np.repeat(np.expand_dims(X,axis=0),centers.shape[0],axis=0)
        completed_iterations = np.zeros(centers.shape[0])
        for _ in range(max_iter):
            dist = np.linalg.norm(X - np.expand_dims(centers,axis=1),axis=-1)
            old_centers = centers
            centers = np.nanmean(np.where(np.expand_dims(dist < radius,axis=-1),X_stack,np.nan),axis=1)
            diff = np.linalg.norm(centers - old_centers,axis =-1)  > stop_thresh
            completed_iterations += diff
            if np.all(~diff):
                break
        n_points_within = np.sum(dist < radius,axis=1)
        return centers,n_points_within,completed_iterations

    def plot(self,X,y_true):
        fig,ax = plt.subplots(1,2,figsize= (10,5))
        ax[0].scatter(X[:,0], X[:,1], s=50, c=self.labels, marker='o')
        ax[0].scatter(self.seeds[:,0], self.seeds[:,1], s=50, c='red', marker='x')
        ax[0].scatter(self.centers[:,0], self.centers[:,1], s=50, c='green', marker='x')
        ax[1].scatter(X[:,0], X[:,1], s=50, c=y_true, marker='o')
        plt.show()

    def fit(self,X):
        bandwidth = self.get_bandwidth(X)
        self.seeds = self.init_centers(X, bandwidth,min_bin_freq=1)
        all_res = self.mean_shift(X,self.seeds,bandwidth,self.max_iter)
        self.n_iter_ = np.max(all_res[2])
        centers,idx = np.unique(all_res[0],axis=0,return_index=True)
        n_points_within = all_res[1][idx]
        sort = np.argsort(n_points_within)[::-1]
        n_points_within = n_points_within[sort]
        centers = centers[sort]
        dist = np.linalg.norm(centers - np.expand_dims(centers, axis=1),axis=-1) > bandwidth
        _,idx = np.unique(dist,return_index=True,axis=0)
        self.centers = centers[idx]
        self.labels = np.argmin(np.linalg.norm(X - np.expand_dims(self.centers, axis=1),axis =-1),axis=0)

x,y = make_blobs(n_samples=1000, centers=3, cluster_std=1.8,random_state=10)

model = Mean_Shift()
model.fit(x)
model.plot(x,y)

print(model.centers)



