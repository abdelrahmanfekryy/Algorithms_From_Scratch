import numpy as np
from matplotlib import pyplot as plt
from HC_model import Hierarchical
from sklearn.datasets import make_circles

x,y = make_circles(n_samples=120, factor=.5, noise=.05,random_state=8)

model = Hierarchical(linkage='single',num_clusters=2)
y_pred1 = model.fit(x)

fig,ax = plt.subplots(1,2,figsize= (10,5))
ax[0].scatter(x[:,0], x[:,1], s=50, c=y_pred1, marker='o')
ax[0].set_title('Hierarchical Clustering')
ax[0].set_xlabel('X1')
ax[0].set_ylabel('X2')
model.dendrogram(ax[1])
ax[1].set_title('Clustering Dendrogram')
ax[1].set_ylabel('Euclidean Distance')
ax[1].set_xlabel('Sample Index')
plt.show()