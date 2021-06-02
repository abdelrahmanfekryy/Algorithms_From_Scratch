<!--Open Preview (Ctrl+Shift+V)-->
# K-Means Clustering:

## Table of Contents
* [Description](#description-)
* [Dependencies](#dependencies-)
* [Implementation Phases](#implementation-phases-)
* [Usage Example](#usage-example-)
* [References](#references-)

## Description :
visualization to explain K-Means clustering phases.

## Dependencies :
* [Numpy](http://www.numpy.org/)
* [Scipy](http://scipy.github.io/devdocs/index.html)
* [Imageio](https://imageio.readthedocs.io/en/stable/)
* [Matplotlib](https://matplotlib.org/)


## Implementation Phases :

*Figure1 created using* `Make_GIF`

<img src='Images/Figure_1.gif' width = 500/>

*Figure2 created using* `Plot_Tracks`

<img src='Images/Figure_2.png' width = 500/>

*Figure3 created using* `Plot_Cluster`

<img src='Images/Figure_3.png' width = 500/>

## Usage Example :
```python
from sklearn.datasets import make_blobs
from KM_model import *
from Plot_assest import *

x,y = make_blobs(n_samples=500, n_features=2, centers=6, cluster_std=1.8,random_state=42)

model = Kmeans(random_state=42,n_clusters=6,init='k-means++',max_iter = 300)
model.fit(x)

Plot_Tracks(x,model)
Plot_Cluster(x,model)
Make_GIF(x,model,dpi=300,f_delay=3)
```


## References :
- **[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)**
- **[wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)**
