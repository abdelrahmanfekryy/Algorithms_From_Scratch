import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

colors = np.array(['#00ff00','#ff0000','#00ffff','#0000ff','#ffff00','#ff00ff','#ff8000','#008000','#800000','#008080','#000080','#808000','#800080','#000000'])

def plot_circle(ax,radius,center=(0,0),resulotion = 100,color='b',linestyle = '-',alpha=1,label=""):
    theta = np.linspace(0, 2*np.pi,resulotion)
    return ax.plot(center[0] + radius*np.cos(theta),center[1] + radius*np.sin(theta),c=color,ls=linestyle,alpha=alpha,label=label)

def Plot_Clusters(model,X):
    plt.scatter(X[:,0], X[:,1], s=30, facecolor='#777777',edgecolor = colors[model.labels_] , marker='o')
    plt.title('DBSCAN Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    legend_elements = [Line2D([0], [0], marker='o', color='None',markeredgecolor=colors[i],markerfacecolor='#777777', markersize=6, label=f'cluster {i}') for i in np.unique(model.labels_) if i>=0]
    legend_elements.append(Line2D([0], [0], marker='o', color='None',markeredgecolor=colors[-1],markerfacecolor='#777777', markersize=6, label='Noise Pt.'))
    plt.legend(title='Info.',handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().set_aspect('equal')
    plt.show()

def Plot_Reachability(model,X):
    plt.scatter(X[:,0], X[:,1], s=30,facecolor='#777777',edgecolor = colors[model.reachability] , marker='o')
    plt.title('DBSCAN Reachability')
    plt.xlabel('X1')
    plt.ylabel('X2')
    pack = zip(['#000000','#ff0000','#00ff00'],['Noise Pt.','Border Pt.','Core Pt.'])
    legend_elements = [Line2D([0], [0], marker='o', color='None',markeredgecolor=c,markerfacecolor='#777777', markersize=6, label=t) for c,t in pack]
    plt.legend(title='Info.',handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
    ax = plt.gca()
    fig = plt.gcf()

    def press(event):
        _,ind = ax.collections[0].contains(event)
        if len(ind["ind"]):
            plt.autoscale(False)
            ax.collections[0].set_facecolor('#777777')
            ax.lines.clear()
            plot_circle(plt,model.eps,ax.collections[0].get_offsets()[ind["ind"][0]],color='k')
            colors = np.array(['#777777','#ffffff'])
            ax.collections[0].set_facecolor(colors[model.neighbors[ind["ind"][0]].astype(int)])
            fig.canvas.draw_idle()
                
    fig.canvas.mpl_connect("button_press_event", press)
    ax.set_aspect('equal')
    plt.show()