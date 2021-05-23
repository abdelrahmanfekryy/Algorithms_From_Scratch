import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150

def Plot_Pre_Shifted_Centers(model,X,grid = 'tri'):
    plt.title('Pre-Shifted Centers:')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.scatter(X[:,0], X[:,1], s=50, c='#aaaaaa', marker='o',edgecolor='#777777',alpha=0.5,zorder=0,label="Data Points")
    plt.scatter(model.seeds[:,0], model.seeds[:,1],s=100, c='#ffffff',edgecolor = '#000000', marker='*',zorder=2,label="initial Centers")
    radius = {'tri':(2*model.bandwidth**2)**0.5,'sqr':model.bandwidth}
    dist = np.linalg.norm(model.seeds - np.expand_dims(model.seeds, axis=1),axis=-1) <= radius[grid]
    for i in range(model.seeds.shape[0]):
        for j in range(i,model.seeds.shape[0]):
            label = f"BandWidth ({np.round(model.bandwidth,2)})" if j == 0 else ""
            if dist[i,j]:
                plt.plot(np.r_[model.seeds[i,0],model.seeds[j,0]],np.r_[model.seeds[i,1],model.seeds[j,1]],c='#000000',ls='-',alpha=0.5,zorder=1,label=label)
    plt.legend(loc=(1.03,0.5))
    plt.gca().set_aspect('equal')
    plt.show()

def Plot_Shifting(model,X):
    plt.title('Shifting Centers:')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.scatter(X[:,0], X[:,1], s=50, c='#aaaaaa', marker='o',edgecolor='#777777',alpha=0.5,zorder=0,label="Data Points")
    plt.scatter(model.seeds[:,0], model.seeds[:,1], s=100, c='#ffffff',edgecolor = '#000000', marker='*',zorder=2,label="initial Centers")
    for i in range(model.centers_history.shape[0]):
        label = "Shifting Tracks" if i == 0 else ""
        plt.plot(model.centers_history[i,:,0],model.centers_history[i,:,1],c='#000000',alpha=0.5,ls='-.',zorder=1,label=label)
    plt.legend(loc=(1.03,0.5))
    plt.gca().set_aspect('equal')
    plt.show()

def plot_circle(center,radius,resulotion,color='b',linestyle = '-',alpha=1,label=""):
    theta = np.linspace(0, 2*np.pi,resulotion)
    return plt.plot(center[0] + radius*np.cos(theta),center[1] + radius*np.sin(theta),c=color,ls=linestyle,alpha=alpha,label=label)


def Plot_Discarding(model,X):
    plt.title('Discard Lower Density Centers Within BW:')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.scatter(X[:,0], X[:,1], s=50, c='#aaaaaa', marker='o',edgecolor='#777777',alpha=0.5,zorder=0,label="Data Points")
    plt.scatter(model.shifted[:,0], model.shifted[:,1], s=100, c='#555555',edgecolor = '#000000', marker='*',zorder=1,label="Lower Density Centers")
    plt.scatter(model.centers[:,0], model.centers[:,1], s=100, c='#ffffff',edgecolor = '#000000', marker='*',zorder=2,label="Higher Density Centers")
    for i in range(model.centers.shape[0]):
        label = "BandWidth Range" if i == 0 else ""
        plot_circle(model.centers[i],model.bandwidth,50,color ='#000000',alpha=0.5,label=label)
    plt.legend(loc=(1.03,0.5))
    plt.gca().set_aspect('equal')
    plt.show()

def Plot_Clustering(model,X):
    plt.title('Nearest Center Clustering:')
    plt.xlabel('X1')
    plt.ylabel('X2')
    colors = np.array(['#ff0000','#00ff00','#00ffff','#0000ff','#ffff00','#ff00ff'])
    for i in np.unique(model.labels):
        Y = X[model.labels == i]
        plt.scatter(Y[:,0], Y[:,1], s=50,facecolors="#777777",edgecolors=colors[i],alpha=0.7, marker='o',zorder=0,label=f"Cluster {i}")
    plt.scatter(model.centers[:,0], model.centers[:,1], s=150, c='#ffffff',edgecolor = '#000000', marker='*',zorder=1,label='Centers')
    plt.legend(loc=(1.03,0.5))
    plt.gca().set_aspect('equal')
    plt.show()
