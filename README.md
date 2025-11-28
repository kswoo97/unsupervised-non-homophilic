# Official implementation of FULE

### Paper information 

- Title: Feature-centric unsupervised node representation learning without homophily assumption
- Authors: Sunwoo Kim, Soo Yong Lee, Kyungho Kim, Hyunjin Hwang, Jaemin Yoo, Kijung Shin.
- Venue: AAAI 2026

### Overview

In this GitHub repository, we provide (1) supplementary material of our paper, and the official code implementation of our proposed method: FULE.
- For supplementary material, refer to the ```supplementary.pdf``` file in the directory.  
- For code, refer to the "how to run" section below.

### Datasets

We support the following datasets.

| Name  | # of nodes | # of edges | # of features | # of classes |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Cora  | 2708 | 5278 | 1433 | 7 |
| Citeseer  | 3327 | 4552 | 3703 | 6 |
| Pubmed  | 19717 | 44324  | 500 | 3 |
| Photo  | 7650 | 119081  | 745 | 8 |
| Computers  | 13752 | 245861  | 767 | 10 |
| Arxiv | 169343 | 1157799  | 128 | 40 |
| Chameleon  | 2277 | 31371 | 2325 | 5 |
| Squirrel  | 5201 | 198353 | 2089 | 5 |
| Actor  | 7600 | 26659 | 932 | 5 |
| Cornell  | 183 | 277 | 1703 | 5 |
| Wisconsin  | 251 | 450 | 1703 | 5 |
| Texas  | 183 | 279 | 1703 | 5 |
| Penn94  | 41554 | 1362229 | 4814 | 3 |
| Flickr  | 89250 | 449878 | 500 | 7 |

In addition, refer to the README.txt file in the ./datasets folder.

### How to run

For node classification, run the following code:
```
Python3 main.py -data Squirrel -device cuda:0 -down classification
```

For clustering, run the following code:
```
Python3 main.py -data Squirrel -device cuda:0 -down clustering
```

Details of each argument are as follows:

- **data** corresponds to the name of the dataset one aims to reproduce. One of {Cora, Citeseer, Pubmed, Photo, Computers, Arxiv, Chameleon, Squirrel, Actor, Cornell, Wisconsin, Texas, Penn94, Flickr} should be given.
- **device** corresponds to the name of the GPU device one aims to use.
- **down** corresponds to the downstream task. One of {classification, clustering} should be given.


### Hyperparameters

Detailed hyperparameter configurations are provided in **FULE_dataloader.py** file.
