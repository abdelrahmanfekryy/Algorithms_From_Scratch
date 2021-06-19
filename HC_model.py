import numpy as np
import matplotlib.pyplot as plt

class Tnode:
    def __init__(self,id_ = None,distance = 0,left = None,right = None):
        self.id = id_
        self.distance = distance
        self.left = left
        self.right = right
        self.x = 0

class Hierarchical:
    def __init__(self,linkage='ward', num_clusters=1):
        func = {'ward': self.ward_link, 
                'complete': self.complete_link,
                'single': self.single_link,
                'centroid': self.centroid_link,
                'average': self.average_link,
                'median': self.median_link}
        self.link_func = func[linkage]
        self._num_clusters = num_clusters

    def Euclidean_Distance(self,p1,p2):
        return np.sqrt(np.sum((p1-p2)**2,axis=-1))

    def update_nodes(self,i,j):
        self.nodes[i].left = Tnode(id_= self.nodes[i].id,distance=self.nodes[i].distance,left = self.nodes[i].left,right =self.nodes[i].right)
        self.nodes[i].right = self.nodes[j]
        self.nodes[i].id = (self.nodes[i].id,self.nodes[j].id)
        self.nodes[i].distance = self.distance_matrix[i,j]
        self.nodes.pop(j)

    def postorderTraversal(self,node):
        if node != None:
            self.postorderTraversal(node.right)
            self.postorderTraversal(node.left)
            if isinstance(node.id, int):
                self.ticks.append(node.id)
                node.x = self.i
                self.i += 1
            else:
                node.x = (node.left.x + node.right.x)/2
                self.xvlines.append([node.left.x,node.left.x])
                self.yvlines.append([node.left.distance,node.distance])
                self.xvlines.append([node.right.x,node.right.x])
                self.yvlines.append([node.right.distance,node.distance])
                self.xhlines.append([node.left.x,node.right.x])
                self.yhlines.append([node.distance,node.distance])
            
    def init_tree(self):
        self.xvlines = []
        self.yvlines = []
        self.xhlines = []
        self.yhlines = []
        self.ticks = []
        self.i = 0
        for node in self.nodes:
            self.postorderTraversal(node)

    def dendrogram(self,ax):
        self.init_tree()
        ax.set_xticks(range(len(self.ticks)))
        ax.set_xticklabels(self.ticks)
        for i in range(len(self.xvlines)):
            ax.plot(self.xvlines[i],self.yvlines[i])
        for i in range(len(self.xhlines)):
            ax.plot(self.xhlines[i],self.yhlines[i])

    def single_link(self,i, j,idx):
        k = np.arange(self.distance_matrix.shape[0])
        k = k[(k != i) & (k != j)]
        minn = np.minimum(self.distance_matrix[i,k], self.distance_matrix[j,k])
        self.distance_matrix[i,k] = minn
        self.distance_matrix[k,i] = minn

    def complete_link(self,i, j,idx):
        k = np.arange(self.distance_matrix.shape[0])
        k = k[(k != i) & (k != j)]
        maxx = np.maximum(self.distance_matrix[i,k], self.distance_matrix[j,k])
        self.distance_matrix[i,k] = maxx
        self.distance_matrix[k,i] = maxx

    def ward_link(self,i, j,idx):
        k  = np.arange(self.distance_matrix.shape[0])
        k = k[(k != i) & (k != j)]
        n_k = np.sum(self.ids == np.expand_dims(idx[k], axis=1),axis=1)
        n_i = len(self.ids[self.ids == idx[i]])
        n_j = len(self.ids[self.ids == idx[j]])
        n_ijk = n_i + n_j + n_k
        new_distance = np.sqrt((n_i+n_k)/n_ijk * self.distance_matrix[i,k]**2 + (n_j+n_k)/n_ijk * self.distance_matrix[j,k]**2 - n_k/n_ijk *self.distance_matrix[i,j]**2)
        self.distance_matrix[i,k] = new_distance
        self.distance_matrix[k,i] = new_distance
    
    def average_link(self,i, j,idx):
        n_i = len(self.ids[self.ids == idx[i]])
        n_j = len(self.ids[self.ids == idx[j]])
        a_i = n_i / (n_i + n_j)
        a_j = n_j / (n_i + n_j)
        k  = np.arange(self.distance_matrix.shape[0])
        k = k[(k != i) & (k != j)]
        new_distance = a_i*self.distance_matrix[i,k] + a_j*self.distance_matrix[j,k]
        self.distance_matrix[i,k] = new_distance
        self.distance_matrix[k,i] = new_distance

    def centroid_link(self,i, j,idx):
        n_i = len(self.ids[self.ids == idx[i]])
        n_j = len(self.ids[self.ids == idx[j]])
        a_i = n_i / (n_i + n_j)
        a_j = n_j / (n_i + n_j)
        b = (n_i * n_j) / (n_i + n_j)**2
        k  = np.arange(self.distance_matrix.shape[0])
        k = k[(k != i) & (k != j)]
        new_distance = np.sqrt(a_i*self.distance_matrix[i,k]**2 + a_j*self.distance_matrix[j,k]**2 - b*self.distance_matrix[i,j]**2)
        self.distance_matrix[i,k] = new_distance
        self.distance_matrix[k,i] = new_distance

    def median_link(self,i, j,idx):
        k  = np.arange(self.distance_matrix.shape[0])
        k = k[(k != i) & (k != j)]
        new_distance =  np.sqrt(0.5 * self.distance_matrix[i,k]**2 + 0.5 * self.distance_matrix[j,k]**2 - 0.25 * self.distance_matrix[i,j]**2)
        self.distance_matrix[i,k] = new_distance
        self.distance_matrix[k,i] = new_distance

    def fit(self,X):
        self.distance_matrix = self.Euclidean_Distance(X,np.expand_dims(X, axis=1))
        np.fill_diagonal(self.distance_matrix, np.inf)
        self.ids = np.arange(X.shape[0])
        idx = np.arange(X.shape[0])
        self.nodes = [Tnode(id_= i) for i in range(X.shape[0])]
        while len(np.unique(self.ids)) > max(self._num_clusters, 1):
            i, j = np.unravel_index(np.argmin(self.distance_matrix), self.distance_matrix.shape)
            self.link_func(i, j, idx)
            self.ids[self.ids == idx[j]] = idx[i]
            self.update_nodes(i,j)
            idx = idx[idx != idx[j]]
            dx = np.arange(self.distance_matrix.shape[0]) != j
            self.distance_matrix = self.distance_matrix[dx,:][:,dx]
        return np.unique(self.ids,return_inverse=True)[1]
