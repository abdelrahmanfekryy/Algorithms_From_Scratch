import numpy as np
from sklearn.datasets import make_blobs
from DBSCAN_model import DBSCAN
from plot_assest import Plot_Clusters,Plot_Reachability

x, y = make_blobs(n_samples=100, centers=3, cluster_std=1.8,random_state=42)

model = DBSCAN(eps=1, minPoints=3)
model.fit(x)

Plot_Clusters(model,x)
Plot_Reachability(model,x)