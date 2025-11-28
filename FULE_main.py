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


import typing
import warnings
import itertools

from torch import Tensor, FloatTensor
from typing import Any, List, Optional, Tuple, Union
from tqdm import trange, tqdm

from FULE_dataloader import * # Dataset loading function

class FULE_STEP1() :
    
    def __init__(self, dataset, device) :
        
        self.X = dataset.x
        self.I1 = torch.eye(dataset.x.shape[0]).to(device)
        self.A11 = to_dense_adj(dataset.edge_index).to(device)[0]
        self.A21 = copy.deepcopy(self.A11 @ self.A11).to(device)
        
        self.A11 = (self.A11/(self.A11.sum(dim = 1).reshape(-1, 1)))
        self.A21 = (self.A21/(self.A21.sum(dim = 1).reshape(-1, 1)))
        
        self.I1 = self.I1 @ self.X
        self.A1 = self.A11 @ self.X
        self.A2 = self.A21 @ self.X
        
        self.device = device
        
    def likelihood(self) :
        W = torch.softmax(self.view_weights, dim = 0) ## Originally sigmoid
        Z = ((W[0] * (self.I1) + W[1] * (self.A1) + W[2] * (self.A2)))
        CVec = (self.centers)
        S = torch.clip(torch.softmax((Z @ (CVec)), dim =1), min = 1e-8) # |V| x C
        
        return -torch.mean(torch.sum(S * torch.log(S), dim = 1)), S, Z
    
    def regularization(self, S) :
        mP = torch.mean(S, dim = 0)
        return (torch.sum((mP * torch.log(mP))))
    
    def matching(self, Z, S) : 
        
        max_values, max_index = torch.max(S, dim = 1)
        
        S_label = torch.zeros_like(S)
        S_label[list(range(S_label.shape[0])), max_index] = 1.0
        S_match = S_label @ S_label.T
        
        Znorm = Z
        ZS = torch.cdist(Znorm, Znorm)
        
        Pos = torch.sum(ZS * S_match)/torch.sum(S_match)
        Neg = torch.sum(ZS * (1 - S_match))/max(((S_match.shape[0] ** 2) - torch.sum(S_match)), 1)
        
        return torch.exp((Pos - Neg))
    
    def fit(self) :
        self.optimizer.zero_grad()
        L_unsuper, S, Z = self.likelihood()
        L_reg = self.regularization(S)
        L_match = self.matching(Z, S)
        L = L_unsuper + L_reg + self.lamda * L_match
        L.backward()
        self.optimizer.step()
        return L.detach().cpu().item()
    
    def train(self, centroid_model, lr, epochs, wd, lamda = 0.1) :
        self.centroid_model = centroid_model
        self.view_weights = centroid_model.view_weights
        self.centers = centroid_model.centers
        self.optimizer = torch.optim.Adam(centroid_model.parameters(), lr = lr, weight_decay = wd)
        self.lamda = lamda
        
        centroid_model.train()
        total_weights = []
        for ep in tqdm(range(epochs)) :
            l = self.fit()
            total_weights.append(l)
            
        return total_weights
    
    @torch.no_grad()
    def returnZ(self) : 
        W = torch.softmax(self.view_weights, dim = 0)
        Z = ((W[0] * (self.I1) + W[1] * (self.A1) + W[2] * (self.A2)))
        return Z
    
class FULE_STEP1_parameters(nn.Module) :
    
    def __init__(self, x, dataset, nC, device) : 
        super(FULE_STEP1_parameters, self).__init__()
        
        self.I1 = torch.eye(dataset.x.shape[0]).to(device)
        
        self.A11 = to_dense_adj(dataset.edge_index).to(device)[0]
        self.A21 = copy.deepcopy(self.A11 @ self.A11).to(device)
        
        self.A11 = (self.A11/(self.A11.sum(dim = 1).reshape(-1, 1)))
        self.A21 = (self.A21/(self.A21.sum(dim = 1).reshape(-1, 1)))
        
        self.A1 = (self.A11)
        self.A2 = (self.A21)
        
        self.X = dataset.x

        self.view_weights = torch.nn.Parameter(torch.tensor([1.0] * 3))

        with torch.no_grad() : 
            TX = ((self.I1 + self.A1 + self.A2)/3) @ (dataset.x)
        
        ## Initialization
        KM = KMeans(n_clusters=nC, random_state=0).fit((TX.cpu().numpy()))
        C = KM.cluster_centers_
        C = torch.tensor(C, dtype = torch.float32).to(device).T
        self.centers = torch.nn.Parameter(C)
        
    def forward(self, x) : ## We don't need a forward pass. This is just a set of parameters
        
        pass
    
class FULE_STEP2(nn.Module) : 
    
    def __init__(self, x, device, dp = 0.1, alpha = 0.5, do_detach = True) : 
        super(FULE_STEP2, self).__init__()
        
        ## Parameters
        self.encoder1 = torch.nn.Linear(x.shape[1], x.shape[1])
        self.layer_norm1 = nn.LayerNorm(x.shape[1])
        self.dropout_layer = torch.nn.Dropout(p = dp)
        
        self.device = device
        self.newx = copy.deepcopy(x)
        self.alpha = alpha
        self.do_detach = do_detach
        
    def forward(self, x) :

        x = self.encoder1(x)
        x = self.layer_norm1(x)
        x = self.dropout_layer(x)
        x = torch.relu(x)

        return x
    
    def gen_loss(self, IDXs, x) :
        
        z_ = self.forward(x)
            
        cur_delta = z_
        newZ = x + cur_delta
        
        if self.do_detach : 
            newZ2 = torch.clone(newZ)
            newZ2 = newZ2.detach()
            S = (torch.cdist(newZ, newZ2)) * self.alpha
        else : 
            S = (torch.cdist(newZ, newZ)) * self.alpha

        Pos = S[IDXs[0], IDXs[1]]
        LN = (S.sum() - Pos.sum() - S.diag().sum()) / (S.shape[0]**2 - len(IDXs[0]) - S.shape[0])
        LP = torch.mean(Pos)
        L1 = torch.exp(LP - LN)
            
        return L1
    
def train_FULE(dataset, fule_step1, fule_step2, lr1, wd1, lamda, do_norm,
               lr2, wd2, alpha, epochs, n_neighs, device) : 

    fule_step1.train()
    Trainer1 = FULE_STEP1(dataset, device)
    L = Trainer1.train(fule_step1, lr = lr1, epochs = epochs, wd = wd1, lamda = lamda)
    with torch.no_grad() : 
        z = Trainer1.returnZ()
        
    if do_norm : 
        z = torch.nn.functional.normalize(z, dim = 1)
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    optimizer = torch.optim.Adam(fule_step2.parameters(), lr = lr2, weight_decay = wd2)
    
    fule_step2.train()
    loss_lists = []
    nearest_neighs = knn_graph(z.cpu().numpy(), n_neighs, metric = "cosine").todense()
    IDXs = np.where(nearest_neighs)
    new_id = [list(IDXs[0]), list(IDXs[1])]
    
    parameters = []
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.999 ** epoch,# 0.999
                                        last_epoch=-1,
                                        verbose=False)
    fule_step2.train()
    for ep in trange(1000) : 
        
        optimizer.zero_grad()
        
        loss = fule_step2.gen_loss(new_id, z)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_lists.append(loss.cpu().detach().item())
        
        if int(ep + 1) % 100 == 0 : 
            
            parameters.append(copy.deepcopy(fule_step2.state_dict()))
            
    return z, parameters, loss_lists

class MLP_Classifier(nn.Module):
    
    def __init__(self, in_channels, hid_channels, out_channels, dp=0.3):
        super(MLP_Classifier, self).__init__()
        
        self.linear1 = nn.Linear(in_channels, hid_channels)
        self.linear2 = nn.Linear(hid_channels, out_channels)
        self.dropout = dp
        
    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x) :
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear2(x)
        return x
    
class ClassifierTrainer():
    
    def __init__(self, graph, train_nodes, val_nodes, test_nodes, device, hidden_dim, 
                model_type):
        self.graph = graph
        self.device = device
        self.in_channels = self.graph.x.size(1)
        self.hid_channels = hidden_dim
        self.out_channels = int(torch.max(self.graph.y).item() + 1)
        self.train_nodes, self.val_nodes, self.test_nodes = train_nodes, val_nodes, test_nodes
        loss_weight = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model_type = model_type
    
    def score(self, graph, model, index_set):
        model.eval()
        
        with torch.no_grad():
            
            if self.model_type == "gnn" : 
                prediction = model(graph.x, graph.edge_index)
            else : 
                prediction = model(graph.x)
                
            val_loss = self.criterion(prediction[index_set], graph.y[index_set])
            _, pred = prediction.max(dim=1)
            true_false = pred[index_set].eq(graph.y[index_set])
            correct = true_false.sum().item()
            acc = correct / len(index_set)
            return acc, val_loss, true_false
        
    def fit(self, graph, model, lr, weight_decay, epochs, early_stop, print_acc = False):
        
        optimizer = torch.optim.Adam(model.parameters(),
                                        lr=lr,
                                        weight_decay=weight_decay)
        step_counter = 0
        self.best_val_acc = 0
        self.best_val_loss = np.inf
        
        for ep in range(epochs):
            model.train()
            optimizer.zero_grad()
            if self.model_type == "gnn" : 
                prediction = model(graph.x, graph.edge_index)
            else : 
                prediction = model(graph.x)
            loss = self.criterion(prediction[self.train_nodes], graph.y[self.train_nodes])
            loss.backward()
            optimizer.step()
            val_acc, val_loss, val_corr = self.score(graph, model, self.val_nodes)
            
            if print_acc : 
                if int(ep + 1 ) % 10 == 0 : 
                    print(val_acc)
            
            if val_acc >= self.best_val_acc :
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                best_model = copy.deepcopy(model)
                step_counter = 0
            else :
                step_counter += 1

            if step_counter > early_stop : 
                break
                
        return best_model
        
    def eval(self, graph, best_model) :
        train_acc, train_loss, train_corr = self.score(graph, best_model, self.train_nodes)
        val_acc, val_loss, val_corr = self.score(graph, best_model, self.val_nodes)
        test_acc, test_loss, test_corr = self.score(graph, best_model, self.test_nodes)
        return train_acc, val_acc, test_acc, test_corr
