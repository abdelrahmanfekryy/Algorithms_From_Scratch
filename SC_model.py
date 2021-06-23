import numpy as np
from sklearn.cluster import KMeans

class Spectral:
    def __init__(self,n_clusters,n_neighbors = 10,random_state = 1,normalize = 'sym',affinity= 'rbf',a=None,b=None,degree=None):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        normfunc = {'sym':self.Symmetric_normalize
                    ,'rw':self.Random_walk_normalize
                    ,None:self.non_normalize}
        self.normfunc = normfunc[normalize]
        affinityfunc = {'rbf':self.rbf_kernel
                        ,'knn':self.nearest_neighbors
                        ,'sqrx':self.squared_exponential_kernel
                        ,'lplc':self.laplacian_kernel}
        if affinity not in affinityfunc.keys():
            raise ValueError("affinity must be 'rbf','knn','sqrx' or 'lplc'")
        self.affinityfunc = affinityfunc[affinity]
        self.a = a

    def euclidean_distance(self,p1,p2):
        return np.sqrt(np.sum((p1-p2)**2,axis=-1))

    def squared_euclidean_distance(self,p1,p2):
        return np.sum((p1-p2)**2,axis=-1)

    def manhattan_distance(self,p1,p2):
        return np.sum(np.abs((p1-p2)),axis=-1)
        
    def nearest_neighbors(self,X):
        distance_matrix = self.euclidean_distance(X,np.expand_dims(X, axis=1))
        idx = np.argsort(distance_matrix,axis = 1)[:,:self.n_neighbors]
        C = np.zeros(distance_matrix.shape)
        np.put_along_axis(C,idx,1,axis=1)
        return (C + C.T)/2

    def squared_exponential_kernel(self,X):
        distance_matrix = self.euclidean_distance(X,np.expand_dims(X, axis=1))
        A = np.sort(distance_matrix,axis = 1)[:,:self.n_neighbors]
        sig = np.mean(A,axis=1)
        sig_ij = np.outer(sig,sig)
        return np.exp(- distance_matrix**2 / (2 * sig_ij))

    def rbf_kernel(self,X):
        return np.exp(-self.a*self.squared_euclidean_distance(X,np.expand_dims(X, axis=1)))

    def laplacian_kernel(self,X):
        return np.exp(-self.a*self.manhattan_distance(X,np.expand_dims(X, axis=1)))

    def Symmetric_normalize(self,A):
        D = np.sum(A,axis=0)
        D_inv_sqrt = np.diag(1/np.where(D == 0, 1, np.sqrt(D)))
        I = np.diag((D != 0).astype(float))
        L_normed = I - D_inv_sqrt @ A @ D_inv_sqrt
        return L_normed,D_inv_sqrt

    def Random_walk_normalize(self,A):
        D = np.sum(A,axis=0)
        D_inv = np.diag(1/np.where(D == 0, 1,D))
        I = np.diag((D != 0).astype(float))
        L_normed = I - D_inv @ A
        return L_normed,D_inv

    def non_normalize(self,A):
        D = np.diag(np.sum(A,axis=0))
        L = D - A
        return L,D

    def fit(self,X):
        self.a = (self.a is None) and (1.0 / X.shape[1]) or self.a
        A = self.affinityfunc(X)
        np.fill_diagonal(A, 0)
        L_normed,D_inv_sqrt = self.normfunc(A) 
        np.fill_diagonal(L_normed, 1)
        _, eigenvcts = np.linalg.eigh(L_normed)
        self.eigenvcts = D_inv_sqrt @ eigenvcts[:,:self.n_clusters]
        model2 = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state)
        model2.fit(self.eigenvcts)
        self.eigencenters = model2.cluster_centers_
        self.labels = model2.labels_





