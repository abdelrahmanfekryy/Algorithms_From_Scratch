from sklearn.datasets import make_blobs
from BIRCH_model import BIRCH
from plot_assest import Plot_BIRCH


x, y = make_blobs(n_samples=300, centers=3, cluster_std=0.4,random_state=0)

model = BIRCH(n_clusters=None,branching_factor=50,threshold=0.6)
model.fit(x)
y_pred = model.predict(x)
Plot_BIRCH(model,x)


model = BIRCH(n_clusters=3,branching_factor=3)
model.fit(x)
y_pred = model.predict(x)
Plot_BIRCH(model,x)