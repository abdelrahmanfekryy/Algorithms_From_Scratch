import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.datasets import make_circles

x,y = make_circles(n_samples=200, factor=.5, noise=.05,random_state=8)

model = AgglomerativeClustering(linkage='single', n_clusters=2)
model.fit(x)
y_pred2 = model.labels_

fig,ax = plt.subplots(1,2,figsize= (10,5))
ax[0].scatter(x[:,0], x[:,1], s=50, c=y_pred2, marker='o')
dendrogram(linkage(x,method='single'),ax=ax[1])
plt.show()