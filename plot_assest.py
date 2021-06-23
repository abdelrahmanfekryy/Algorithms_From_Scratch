import numpy as np
import matplotlib.pyplot as plt

def Plot_Spectral(model,X):
    fig,ax = plt.subplots(1,2,figsize= (10,5))
    ax[0].set_title('EigenVectors K_Means Clustering')
    ax[0].set_xlabel('EigenVector1')
    ax[0].set_ylabel('EigenVector2')
    ax[0].set_xticklabels([])
    ax[0].scatter(model.eigencenters[:,0], model.eigencenters[:,1], s=100, facecolor='w',edgecolor='k', marker='*',zorder = 1)
    ax[0].scatter(model.eigenvcts[:,0], model.eigenvcts[:,1], s=50, c=model.labels, marker='o',zorder = 0)
    ax[1].set_title('Spectral Clustering')
    ax[1].set_xlabel('X1')
    ax[1].set_ylabel('X2')
    ax[1].scatter(X[:,0], X[:,1], s=50, c=model.labels, marker='o')
    plt.show()