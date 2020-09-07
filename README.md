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
*   -m / --method METHOD: a basic 'ae', 'gae', or 'vgae', Graph AutoEncoder or Variational Graph AutoEncoder 
*   -v / --visualize : false if omitted, true otherwise: for rendering embeddings (only available for 'cora' dataset)
*   -dv / --device DEVICE: 'gpu' or 'cpu'. If cuda is not available, it will run on CPU by default. 
*   -f / --features : false if omitted, true otherwise: use node features (only available for 'cora' and 'citeseer' datasets)
*   -k / --kcross : false if omitted, true otherwise: run 10 training to compute cross validation on different splits.
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
*   -k / --kcross : false if omitted, true otherwise: run 10 training to compute cross validation on different splits.
---
### Experimental results (AUC %):

|                  |   Cora    |   Citeseer    |   Facebook    |   PubMed  |
|------------------|-----------|---------------|---------------|-----------|
|     **AE***      |    87.2   |   82.5        |      -        |    -      |
|     **GAE***     |    89.5   |   85.6        |      -        |    -      |
|     **VGAE***    |    90.9   |   85.8        |      -        |    -      |
|     **AE**       |    85.4   |   78.5        |     98.7      |           |
|     **GAE**      |    85.9   |   78.3        |     99.0      |   85.5    |
|     **VGAE**     |    84.6   |   77.0        |     98.7      |   81.9    |
|     **DVNE**     |    90.8   |   86.8        |     98.7      |   86.0    |

_* results using node features when available_

---
