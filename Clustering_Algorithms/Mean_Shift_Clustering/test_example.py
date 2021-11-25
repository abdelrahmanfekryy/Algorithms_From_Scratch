from sklearn.datasets import make_blobs
from MS_model import *
from plot_assest import *

x,y = make_blobs(n_samples=1000, centers=3, cluster_std=1.8,random_state=38)

model = Mean_Shift()
model.fit(x)
Plot_Pre_Shifted_Centers(model,x,grid='sqr')
Plot_Shifting(model,x)
Plot_Discarding(model,x)
Plot_Clustering(model,x)
