import torch
import copy
import numpy as np
import random

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph as knn_graph
from sklearn import preprocessing

import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm

import typing
import warnings
import itertools

from torch import Tensor, FloatTensor
from typing import Any, List, Optional, Tuple, Union

from FULE_dataloader import * # Dataset loading function
import faiss

class FULE_STEP1_large() :
    
    def __init__(self, centroid_model, device) :
        
        self.centroid_model = centroid_model
        
        self.device = device
        
    def likelihood(self) :
        
        Z = self.centroid_model()
        CVec = (self.centers)
        S = torch.clip(torch.softmax((Z @ (CVec)), dim =1), min = 1e-8) # |V| x C
        
        return -torch.mean(torch.sum(S * torch.log(S), dim = 1)), S, Z
    
    def regularization(self, S) :
        
        mP = torch.mean(S, dim = 0)
        
        return (torch.sum((mP * torch.log(mP))))
    
    def fit(self) :
        
        self.optimizer.zero_grad()
        L_unsuper, S, Z = self.likelihood()
        L_reg = self.regularization(S)
        L = L_unsuper + L_reg
        L.backward()
        self.optimizer.step()
        
        return L.detach().cpu().item()
    
    def train(self, lr, epochs, wd) :
        
        self.view_weights = self.centroid_model.view_weights
        self.centers = self.centroid_model.centers
        self.optimizer = torch.optim.Adam(self.centroid_model.parameters(), lr = lr, weight_decay = wd)
        
        self.centroid_model.train()
        total_loss = []
        for ep in tqdm(range(epochs)) :
            l = self.fit()
            total_loss.append(l)
            
        return total_loss
    
    @torch.no_grad()
    def returnZ(self) : 
        Z = self.centroid_model()
        return Z

class FULE_STEP1_large_parameters(nn.Module) :
    
    def __init__(self, X0, X1, X2, nC, device) : 
        super(FULE_STEP1_large_parameters, self).__init__()
        
        self.I1 = X0
        self.A1 = X1
        self.A2 = X2
        
        dim = self.I1.shape[1]
        
        self.view_weights = torch.nn.Parameter(torch.tensor([1.0] * 3))
        
        self.mapper = torch.nn.Sequential(torch.nn.Linear(X1.shape[1], dim, bias = True),  ## Best one was originally layer-1
                                          torch.nn.BatchNorm1d(dim),
                                          torch.nn.Tanh(),
                                         ).to(device)

        with torch.no_grad() : 
            TX = self.mapper((self.I1 + self.A1 + self.A2)/3)
            VS = list(np.random.choice(TX.shape[0], size = nC, replace = False))
            C = torch.tensor(copy.deepcopy((TX[VS, :])).T, dtype = torch.float32).to(device)

        self.centers = torch.nn.Parameter(C)
        
    def forward(self) : 
        
        W = torch.softmax(self.view_weights, dim = 0) ## Originally sigmoid
        Z = self.mapper((W[0] * (self.I1) + W[1] * (self.A1) + W[2] * (self.A2)))

        return Z
    
    
class NN_large() : 
    
    def __init__(self, z, maxK) : 
        
        # Based on FAISS
        X = z.cpu().numpy()
        X = X.astype(np.float32)
        d = X.shape[1]  
        index = faiss.IndexFlatL2(d)
        index.add(X)
        self.maxK = maxK

        _, self.indices = index.search(X, maxK + 1)

        self.indices = self.indices[:, 1:]
        self.n_nodes = X.shape[0]
        
    def slicing(self) : 
        
        cur_indices = self.indices
        cur_indices = cur_indices.flatten()
        rows = []

        for i in range(self.n_nodes) : 
            rows.extend([i] * self.maxK)

        new_id = [rows, list(cur_indices)]
    
        return new_id
    
class FULE_STEP2_large(nn.Module) : 
    
    def __init__(self, x, device, dp = 0.1, alpha = 0.5) : 
        super(FULE_STEP2_large, self).__init__()
        
        self.encoder1 = torch.nn.Linear(x.shape[1], x.shape[1])
        self.layer_norm1 = nn.LayerNorm(x.shape[1])
        self.dropout_layer = torch.nn.Dropout(p = dp)
        self.device = device
        
        self.newx = copy.deepcopy(x)
        
        self.alpha = alpha
        
        
    def forward(self, x) : ## We are learning deltas
        
        x = self.encoder1(x)
        x = self.layer_norm1(x)
        x = self.dropout_layer(x)
        x = torch.tanh(x)
        
        return x
    
    def gen_loss(self, IDXs, x) :
        
        z_ = self.forward(x)
            
        newZ = x + z_

        LP = torch.mean(torch.sqrt((torch.sum((newZ[IDXs[0]] - newZ[IDXs[1]])**2, dim = 1) + 1e-8))) * self.alpha 
        neg_idxs = list(np.random.choice(newZ.shape[0], size = 100, replace = False)) # Negative sampling
        LN = (torch.cdist(newZ, newZ[neg_idxs]) * self.alpha).mean()

        L1 = torch.exp(LP - LN)

        return L1
    
def train_FULE_large(dataset, fule_step1, fule_step2, lr1, wd1, 
               lr2, wd2, n_neighs, epochs, device) :  # Since nearest neighbor loading takes large amount of time, we pre-compute the neighbors
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    fule_step1.train()
    fule_step2.train()
    
    step1_trainer = FULE_STEP1_large(fule_step1, device)
    loss_lists1 = step1_trainer.train(lr = lr1, epochs = 1000, wd = wd1)
    
    with torch.no_grad() : 
        z = step1_trainer.returnZ()
    
    loss_lists = []
    
    neigh_lists = NN_large(z = z, maxK = n_neighs).slicing()
    
    model = fule_step2
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr2, 
                                 weight_decay = wd2)
    
    newz = copy.deepcopy(z)
    parameters = []
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.999 ** epochs, # 0.999
                                        last_epoch=-1,
                                        verbose=False)
    
    for ep in (range(500)) : 
        
        model.train()
        optimizer.zero_grad()
        
        loss = model.gen_loss(neigh_lists, newz)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_lists.append(loss.cpu().detach().item())

        if int(ep + 1) % 100 == 0 : 
            
            parameters.append(copy.deepcopy(model.state_dict()))
        
    return z, parameters, loss_lists
