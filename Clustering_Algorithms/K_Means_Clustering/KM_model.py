import numpy as np

class Kmeans:
    def __init__(self,n_clusters = 2,random_state = 1,init='k-means++',max_iter = 100,tolerance = 1e-4):
        self.n_clusters=n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.tolerance = tolerance
        init_func = {'k-means++':self.kmpp,'random':self.random}
        self.init_func = init_func[init]

    def squared_euclidean_distance(self,p1,p2):
        return np.sum((p1-p2)**2,axis=-1)

    def random(self,X,n_clusters,random_state):
        n_samples = X.shape[0]
        idxs = np.random.RandomState(random_state).permutation(n_samples)[:n_clusters]
        return X[idxs]

    def kmpp(self,X,n_clusters,random_state):
        idx = np.random.RandomState(random_state).randint(X.shape[0])
        centers = [X[idx]]
        for _ in range(1, n_clusters):
            dist_sq = self.squared_euclidean_distance(X,np.expand_dims(centers,axis=1))
            dist_sq = np.min(dist_sq,axis=0)
            probs = dist_sq/np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = np.random.RandomState(random_state).rand()
            idx = np.searchsorted(cumulative_probs,r)
            centers.append(X[idx])

        return np.array(centers)

    def WCSS(self,X,centers,labels):
        '''Within Cluster Sum of Squares'''
        return np.sum((X - centers[labels])**2)

    def Lloyd_Algorithm(self,X,centers,tolerance,max_iter):
        n_clusters = centers.shape[0]
        n_iter = 0
        centers_history = np.expand_dims(centers,axis=0)
        labels_history = np.zeros((0,X.shape[0]),int)
        for _ in range(max_iter):
            n_iter += 1
            dist_sq = self.squared_euclidean_distance(X,np.expand_dims(centers, axis=1))
            labels = np.argmin(dist_sq,axis=0)
            labels_history = np.append(labels_history,np.expand_dims(labels,axis=0),axis=0)
            centers = np.array([np.mean(X[labels == i],axis=0) for i in range(n_clusters)])
            if np.all(np.abs(centers_history[-1] - centers) <= tolerance):
                break
            centers_history = np.vstack((centers_history,np.expand_dims(centers,axis=0)))

        return centers_history,labels_history,n_iter,

    def fit(self,X):
        self.init_centers = self.init_func(X,self.n_clusters,self.random_state)
        Data = self.Lloyd_Algorithm(X,self.init_centers,self.tolerance,self.max_iter)
        self.centers = Data[0][-1]
        self.centers_history = np.transpose(Data[0],axes=(1,0,2))
        self.labels = Data[1][-1]
        self.labels_history = Data[1]
        self.n_iter = Data[2]
        self.inertia = self.WCSS(X,self.centers,self.labels)

    def predict(self,X):
        dist_sq = self.squared_euclidean_distance(X,np.expand_dims(self.centers, axis=1))
        return np.argmin(dist_sq,axis=0)


