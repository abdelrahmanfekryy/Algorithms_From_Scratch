import numpy as np
from sklearn.cluster import AgglomerativeClustering

############################################################################################
class _CFSubcluster:
    def __init__(self,idx=None,Samples=None):
        self.idxs = []
        self.N = 0
        self.SS = 0
        self.LS = 0
        self.child_ = None

        if idx is not None:
            self.idxs.append(idx)
        if Samples is not None:
            self.N = 1
            self.LS = Samples
            self.SS = Samples @ Samples
  
    def get_Centroid(self):
        return self.LS/self.N

    def get_Radius(self):
        return np.sqrt(max(0, (self.SS/self.N) - (self.LS/self.N) @ (self.LS/self.N)))

    def get_Average_Distance_BC(self,subcluster):
        return ((self.LS/self.N) @ (self.LS/self.N) - 2.0 * (self.LS/self.N)  @ (subcluster.LS/subcluster.N))
        
###########################################################################################
class _CFNode:
    def __init__(self,threshold, branching_factor, is_leaf):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.subclusters_ = []
        self.prev_leaf_ = None
        self.next_leaf_ = None

    def squared_euclidean_distance(self,p1,p2):
        return np.sum((p1-p2)**2,axis=-1)

    def split(self):
        new_subcluster1 = _CFSubcluster()
        new_subcluster2 = _CFSubcluster()
        new_node1 = _CFNode(threshold=self.threshold,branching_factor=self.branching_factor,is_leaf=self.is_leaf)
        new_node2 = _CFNode(threshold=self.threshold,branching_factor=self.branching_factor,is_leaf=self.is_leaf)
        new_subcluster1.child_ = new_node1
        new_subcluster2.child_ = new_node2

        if self.is_leaf:
            new_node1.prev_leaf_ = self.prev_leaf_
            new_node1.next_leaf_ = new_node2
            new_node2.prev_leaf_ = new_node1
            new_node2.next_leaf_ = self.next_leaf_
            if self.next_leaf_:
                self.next_leaf_.prev_leaf_ = new_node2
            if self.prev_leaf_:
                self.prev_leaf_.next_leaf_ = new_node1

        centroids = np.array([sub.get_Centroid() for sub in self.subclusters_])
        dist = self.squared_euclidean_distance(centroids,np.expand_dims(centroids,axis=1))
        
        farthest_idx = np.unravel_index(np.argmax(dist),dist.shape)
        node1_dist, node2_dist = dist[farthest_idx,:]

        for idx, subcluster in enumerate(self.subclusters_):
            if node1_dist[idx] < node2_dist[idx]:
                new_node1.subclusters_.append(subcluster)
                self.merge_subcluster(new_subcluster1,subcluster)
            else:
                new_node2.subclusters_.append(subcluster)
                self.merge_subcluster(new_subcluster2,subcluster)
        return new_subcluster1, new_subcluster2

    def merge_subcluster(self,subcluster1,subcluster2):
        subcluster1.N += subcluster2.N
        subcluster1.LS += subcluster2.LS
        subcluster1.SS += subcluster2.SS
        subcluster1.idxs += subcluster2.idxs

    def insert_subcluster(self, subcluster):
        if not self.subclusters_:
            self.subclusters_.append(subcluster)
            return False

        closest_index = np.argmin([sub.get_Average_Distance_BC(subcluster) for sub in self.subclusters_])

        if self.subclusters_[closest_index].child_:
            split_child = self.subclusters_[closest_index].child_.insert_subcluster(subcluster)
            if split_child:
                new_subcluster1, new_subcluster2 = self.subclusters_[closest_index].child_.split()
                self.subclusters_[closest_index] = new_subcluster1
                self.subclusters_.append(new_subcluster2)
            else:
                self.merge_subcluster(self.subclusters_[closest_index],subcluster)

        else:
            SS = self.subclusters_[closest_index].SS + subcluster.SS
            LS = self.subclusters_[closest_index].LS + subcluster.LS
            N = self.subclusters_[closest_index].N + subcluster.N
            radius = np.sqrt((SS / N) - (LS / N) @ (LS / N))
            if radius <= self.threshold:
                self.merge_subcluster(self.subclusters_[closest_index],subcluster)
                
            else:
                self.subclusters_.append(subcluster)

        return len(self.subclusters_) > self.branching_factor
###########################################################################################

class BIRCH:
    def __init__(self,threshold=0.5,branching_factor=50,n_clusters=None):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters

    def euclidean_distance(self,p1,p2):
        return np.sqrt(np.sum((p1-p2)**2,axis=-1))

    def fit(self, XX):
        X = XX.copy()
        self.root_ = _CFNode(threshold=self.threshold,branching_factor=self.branching_factor,is_leaf=True)
        self.dummy_leaf_ = _CFNode(threshold=self.threshold,branching_factor=self.branching_factor,is_leaf=True)
        self.dummy_leaf_.next_leaf_ = self.root_
        self.root_.prev_leaf_ = self.dummy_leaf_

        for idx,sample in enumerate(X):
            subcluster = _CFSubcluster(idx=idx,Samples=sample)
            split = self.root_.insert_subcluster(subcluster)

            if split:
                new_subcluster1, new_subcluster2 = self.root_.split()
                del self.root_
                self.root_ = _CFNode(threshold=self.threshold,branching_factor=self.branching_factor,is_leaf=False)
                self.root_.subclusters_.append(new_subcluster1)
                self.root_.subclusters_.append(new_subcluster2)
        
        
        self.subcluster_centers = []
        node = self.dummy_leaf_.next_leaf_
        while node:
            self.subcluster_centers += [sub.get_Centroid() for sub in node.subclusters_]
            node = node.next_leaf_
        
        self.subcluster_centers = np.array(self.subcluster_centers)

        if self.n_clusters:
            model2 = AgglomerativeClustering(n_clusters=self.n_clusters)
            self.subcluster_labels_ = model2.fit_predict(self.subcluster_centers)
        else:
            self.subcluster_labels_ = np.arange(len(self.subcluster_centers))

    def predict(self, X):
        minidxs = np.argmin(self.euclidean_distance(X,np.expand_dims(self.subcluster_centers, axis=1)),axis=0)
        return self.subcluster_labels_[minidxs]
