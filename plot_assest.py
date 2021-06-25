import numpy as np
import matplotlib.pyplot as plt

color = '#0066cc'
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['text.color'] = color
plt.rcParams["legend.edgecolor"] = color
plt.rcParams['figure.facecolor'] = '#22222200'
plt.rcParams['axes.edgecolor'] = color
plt.rcParams['axes.labelcolor'] = color
plt.rcParams['axes.titlecolor'] = color
plt.rcParams['xtick.color'] = color
plt.rcParams['ytick.color'] = color
plt.rcParams['legend.borderpad'] = 0.3

def Plot_BIRCH(model,X):
    plt.title('BIRCH Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.scatter(model.subcluster_centers[:,0], model.subcluster_centers[:,1], s=100, facecolor='w',edgecolor='k', marker='*',zorder = 1)
    plt.scatter(X[:,0], X[:,1], s=50, c=model.predict(X), marker='o',zorder = 0)
    plt.gca().set_aspect('equal')
    plt.show()