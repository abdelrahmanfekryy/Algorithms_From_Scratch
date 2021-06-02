import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,voronoi_plot_2d

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

def Plot_Cluster(X,model):
    plt.title('K_Means Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    colors = np.array(['#ff0000','#00ff00','#00ffff','#0000ff','#ffff00','#ff00ff'])
    for i in np.unique(model.labels):
        Y = X[model.labels == i]
        plt.scatter(Y[:,0],Y[:,1], s=50,facecolors="#777777",edgecolors=colors[i],alpha=0.7,marker='o',zorder=1,label=f"Cluster {i}")
    plt.scatter(model.centers[:,0], model.centers[:,1], s=150, c='#ffffff',edgecolor = '#000000', marker='*',zorder=2,label='Centers')
    plt.legend(title='Info.',loc='center left', bbox_to_anchor=(1, 0.5))
    X1_min,X1_max = plt.gca().get_xlim()
    X2_min,X2_max = plt.gca().get_ylim()
    vor = Voronoi(model.centers)
    voronoi_plot_2d(vor,ax=plt.gca(),show_points=False,show_vertices=False)
    plt.gca().set_xlim(X1_min,X1_max)
    plt.gca().set_ylim(X2_min,X2_max)
    plt.gca().set_aspect('equal')
    plt.show()

def Plot_Tracks(X,model):
    plt.title('Centers tracks:')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.scatter(X[:,0], X[:,1], s=50, c='#aaaaaa', marker='o',edgecolors='#777777',alpha=0.5,zorder=0,label="Data Point")
    plt.scatter(model.init_centers[:,0],model.init_centers[:,1], s=150,facecolors='#ffffff',edgecolors = '#000000', marker='*',zorder=2,label="Init. Seed")
    for i in range(model.centers_history.shape[0]):
        label = "Tracks" if i == 0 else ""
        plt.plot(model.centers_history[i,:,0],model.centers_history[i,:,1],c='#000000',ls='-',marker='o',markersize=3,zorder=1,label=label)
    plt.legend(title='Info.',loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().set_aspect('equal')
    plt.show()

def Make_GIF(X,model,dpi=100,f_delay = 3):
    plt.gcf().set_dpi(dpi)
    plt.xlabel('X1')
    plt.ylabel('X2')
    colors = np.array(['#ff0000','#00ff00','#00ffff','#0000ff','#ffff00','#ff00ff'])
    dir_path = os.path.dirname(os.path.realpath(__file__))
    writer = imageio.get_writer(dir_path + '/PIC.gif', mode='I')
    plt.gca().set_aspect('equal')
    for i in range(model.centers_history.shape[1]):
        plt.gca().collections.clear()
        plt.title(f'K_Means Clustering, iter= {i}')
        for j in np.unique(model.labels):
            Y = X[model.labels_history[i] == j]
            plt.scatter(Y[:,0], Y[:,1], s=50, marker='o',facecolors='#777777',edgecolors =colors[j],label=f"Cluster {j}",alpha=0.5)
        plt.scatter(model.centers_history[:,i,0],model.centers_history[:,i,1],s=150,facecolors='#ffffff',edgecolors = '#000000',marker='*',label="Centers")
        X1_min,X1_max = plt.gca().get_xlim()
        X2_min,X2_max = plt.gca().get_ylim()
        voronoi_plot_2d(Voronoi(model.centers_history[:,i]),ax=plt.gca(),show_points=False,show_vertices=False)
        plt.legend(title='Info.',loc='center left', bbox_to_anchor=(1, 0.5))
        plt.gca().set_xlim(X1_min,X1_max)
        plt.gca().set_ylim(X2_min,X2_max)
        plt.gcf().canvas.draw()
        buffer = np.array(plt.gcf().canvas.renderer.buffer_rgba(),dtype='uint8')
        for _ in range(f_delay):
            writer.append_data(buffer)
    plt.close()