from sklearn.datasets import make_blobs
from KM_model import *
from Plot_assest import *

x,y = make_blobs(n_samples=500, n_features=2, centers=6, cluster_std=1.8,random_state=42)

model = Kmeans(random_state=42,n_clusters=6,init='k-means++',max_iter = 300)
model.fit(x)

Plot_Tracks(x,model)
Plot_Cluster(x,model)
Make_GIF(x,model,dpi=300,f_delay=3)