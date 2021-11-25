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

colors = np.array(['#ff0000','#00ff00','#00ffff','#0000ff','#ffff00','#ff00ff','#000000'])

def Plot_OPTICS(model,X):
    plt.scatter(X[:,0], X[:,1], s=30, c='#80808080',edgecolor = colors[model.labels_] , marker='o')
    plt.title('OPTICS Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    handles=[plt.Line2D([0], [0], marker='o',color='None',markerfacecolor='#80808080',markeredgecolor=colors[label],label=f'cluster {label}' if label != -1 else 'Noise Pt.') for label in np.unique(model.labels_)]
    plt.legend(title='Info.',handles=handles)
    plt.gca().set_aspect('equal')
    plt.show()

def Reachability_Plot(model):
    plt.gcf().set_size_inches(10, 5)
    plt.grid(axis='y', alpha=0.75)
    plt.bar(x=np.arange(model.labels_.shape[0]),height=model.reachability_[model.ordering_],color='#808080',edgecolor=colors[model.labels_[model.ordering_]])
    plt.title('Reachability plot')
    plt.ylabel('Reachability (epsilon distance)')
    handles=[plt.Rectangle((0,0),1,1, facecolor='#80808080',edgecolor=colors[label],label=f'cluster {label}' if label != -1 else 'Noise Pt.',lw=2) for label in np.unique(model.labels_)]
    plt.legend(title='Info.',handles=handles)
    plt.show()