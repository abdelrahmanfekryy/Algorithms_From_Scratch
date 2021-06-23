import numpy as np
from sklearn.datasets import make_circles
from SC_model import Spectral
from plot_assest import Plot_Spectral


x,y = make_circles(n_samples=300, factor=.5, noise=.05,random_state=8)

model = Spectral(n_clusters=2,random_state=42, n_neighbors=20,affinity='knn')
model.fit(x)
Plot_Spectral(model,x)