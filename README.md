# Neural Graph Autoencoders for Link Reconsturction
###### Machine Learning Course Project 19/20 - University of Florence

The code is an implementation of material from these papers:

* [Semi-supervised Classification for Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)
* [Variational Graph Auto-Encoders](https://arxiv.org/pdf/1611.07308.pdf)
* [Deep Variational Network Embedding in Wasserstein Space](https://dl.acm.org/doi/pdf/10.1145/3219819.3220052)

The purpose of these autoencoders is the reconstruction of missing edges from the training graphs.

#### Requirements:
* Python 3.7
* pytorch (cuda availability is optional)
* SciKit-Learn
* NumPy
* MatPlotLib
* h5Py
* 8 GB RAM/VRAM (32GB for PUBMED dataset!)

---

##### For running the code (GAE/VGAE):
```
$ python train_gae.py -m METHOD -d DATASET --device DEVICE \
-f USE_FEATURES -v VISUALIZE
``` 
e.g.: `$ python train_gae.py -m vgae -d citeseer -d gpu`

##### Options: 
*   -d / --dataset DATASET: one between 'cora', 'citeseer', 'facebook', 'pubmed' 
*   -m / --method METHOD: 'gae', or 'vgae', Graph AutoEncoder or Variational Graph AutoEncoder 
*   -v / --visualize VISUALIZE: false if omitted, true otherwise: for rendering embeddings (only available for 'cora' dataset)
*   -dv / --device DEVICE: 'gpu' or 'cpu'. If cuda is not available, it will run on CPU by default. 
*   -f / --features FEATURES: false if omitted, true otherwise: use node features (only available for 'cora' and 'citeseer' datasets)
---
##### For running the code (DVNE):
```
$ python train_dvne.py -d DATASET --device DEVICE -v VISUALIZE
``` 
e.g.: `$ python train_gae.py -d cora -dv gpu`

##### Options: 
*   -d / --dataset DATASET: one between 'cora', 'citeseer', 'facebook', 'pubmed' 
*   -v / --visualize VISUALIZE: false if omitted, true otherwise: for rendering embeddings (only available for 'cora' dataset)
*   -dv / --device DEVICE: 'gpu' or 'cpu'. If cuda is not available, it will run on CPU by default. 
---
### Experimental results (AUC %):

|                  |   Cora    |   Citeseer    |   Facebook    |   PubMed  |
|------------------|-----------|---------------|---------------|-----------|
|     **GAE***     |    88.8   |   84.6        |      -        |    -      |
|     **VGAE***    |    90.1   |   85.6        |      -        |    -      |
|     **GAE**      |    85.2   |   79.9        |           |   85.5    |
|     **VGAE**     |    84.1   |   77.7        |           |   81.3    |
|     **DVNE**     |    90.7   |           |           |   85.7    |    

_* results using node features when available_

---