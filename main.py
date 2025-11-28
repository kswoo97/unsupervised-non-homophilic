import torch
import copy
import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score 

from FULE_dataloader import *
from FULE_main import *
from FULE_large import * 

if __name__ == "__main__" : 
    
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser('Proposed Method: FULE.')
    
    parser.add_argument('-data', '--data', type=str, default='Cora')            
    parser.add_argument('-device', '--device', type=str, default='cuda:0')
    parser.add_argument('-down', '--down', type=str, default='classification')
    
    args = parser.parse_args()
    
    d_name = args.data
    device = args.device
    downstream = args.down
    
    if d_name in ["Arxiv", "Flickr", "Penn94"] : 
        dataset, data_x, lr1, wd1, lamda, epochs1, lr2, wd2, K, dp, tau, do_detach, do_norm, train_idxs, valid_idxs, test_idxs = prepare_dataset_and_hyperparameter_config_large (d_name, downstream, device)
        
        gt_labels = dataset.y.cpu().numpy()
        
        parformance_table1 = np.zeros((10, 5))
        parformance_table2 = np.zeros((10, 5))
        
        X0 = data_x[0].to(device)
        X1 = data_x[1].to(device)
        X2 = data_x[2].to(device)
        
        X0 = torch.nn.functional.normalize(X0, dim = 1)
        X1 = torch.nn.functional.normalize(X1, dim = 1)
        X2 = torch.nn.functional.normalize(X2, dim = 1)
        
        for i in range(10) : 
            
            torch.manual_seed(i)
            fule_step1 = FULE_STEP1_large_parameters(X0, X1, X2, nC = dataset.y.unique().shape[0], device = device).to(device)
            torch.manual_seed(i)
            fule_step2 = FULE_STEP2_large(device = device, x = X0, dp = dp, alpha = tau).to(device)
            
            z, parameters, losses = train_FULE_large(dataset = dataset, fule_step1 = fule_step1, fule_step2 = fule_step2, 
                                                     lr1 = lr1, wd1 = wd1, 
                                                     lr2 = lr2, wd2 = wd2, n_neighs = K, epochs = 500, device = device)
            
            if len(train_idxs) > 1 : # More than 5 splits
                cur_train = train_idxs[i]
                cur_valid = valid_idxs[i]
                cur_test = test_idxs[i]
            
            else : # Single split
                cur_train = train_idxs[0]
                cur_valid = valid_idxs[0]
                cur_test = test_idxs[0]

            for idid, param in enumerate(parameters) : 
                
                prevZ = copy.deepcopy(z)
                newD = copy.deepcopy(dataset)
                
                fule_step2.load_state_dict(param)
                with torch.no_grad() : 
                    fule_step2.eval()
                    newZ = prevZ + fule_step2(prevZ)
            
                newD.x = newZ

                dim = newZ.shape[1]

                torch.manual_seed(i)

                if downstream == "classification" :  # Node Classification

                    mlp_classifier = MLP_Classifier(in_channels = newD.x.shape[1], hid_channels = 1024, 
                                                    out_channels = dataset.y.unique().shape[0], 
                                                    dp = 0.3).to(device)

                    trainer = ClassifierTrainer(graph = newD, train_nodes = cur_train, 
                                                val_nodes = cur_valid, test_nodes = cur_test,
                                                device = device, hidden_dim = dim, model_type = "mlp")


                    mlp_classifier = trainer.fit(newD, model = mlp_classifier, lr = 1e-3, 
                                                 weight_decay = 1e-6, epochs = 1000, early_stop = 500)

                    train_acc, valid_acc, test_acc, _ = trainer.eval(newD, mlp_classifier)
                    parformance_table1[i, idid] = valid_acc
                    parformance_table2[i, idid] = test_acc

                else : ## Clustering

                    kmeans = KMeans(n_clusters=dataset.y.unique().shape[0],  
                                    random_state=i, n_init=10).fit(newZ.cpu().numpy())
                    pred_labels = kmeans.labels_

                    valid_acc = nmi_score(gt_labels[cur_valid], pred_labels[cur_valid])
                    test_acc = nmi_score(gt_labels, pred_labels)

                    parformance_table1[i, idid] = valid_acc
                    parformance_table2[i, idid] = test_acc
            
            print("Seed:", i)
            print("Validation:", parformance_table1[i, :])
            print("Test:", parformance_table2[i, :])
            print()
        
    else : 
        dataset, lr1, wd1, lamda, epochs1, lr2, wd2, K, dp, tau, do_detach, do_norm, train_idxs, valid_idxs, test_idxs = prepare_dataset_and_hyperparameter_config_medium (d_name, downstream, device)
        
        gt_labels = dataset.y.cpu().numpy()
    
        parformance_table1 = np.zeros((10, 10))
        parformance_table2 = np.zeros((10, 10))

        for i in range(10) : 

            torch.manual_seed(i)
            fule_step1 = FULE_STEP1_parameters(x = dataset.x, dataset = dataset, nC = dataset.y.unique().shape[0], device = device).to(device)
            torch.manual_seed(i)
            fule_step2 = FULE_STEP2(x = dataset.x, device = device, dp = dp, alpha = tau, do_detach = do_detach).to(device)

            Z, parameters, losses = train_FULE(dataset = dataset, fule_step1 = fule_step1, fule_step2 = fule_step2, 
                                               lr1 = lr1, wd1 = wd1, lamda = lamda, do_norm = do_norm,
                                               lr2 = lr2, wd2 = wd2, alpha = tau, epochs = epochs1, n_neighs = K, device = device)
            
            if len(train_idxs) > 1 : # More than 5 splits
                cur_train = train_idxs[i]
                cur_valid = valid_idxs[i]
                cur_test = test_idxs[i]
            
            else : # Single split
                cur_train = train_idxs[0]
                cur_valid = valid_idxs[0]
                cur_test = test_idxs[0]

            for idid, param in enumerate(parameters) : 
                
                prevZ = copy.deepcopy(Z)
                newD = copy.deepcopy(dataset)
                
                fule_step2.load_state_dict(param)
                with torch.no_grad() : 
                    fule_step2.eval()
                    newZ = prevZ + fule_step2(prevZ)
            
                newD.x = newZ

                dim = newZ.shape[1]

                torch.manual_seed(i)

                if downstream == "classification" :  # Node Classification

                    mlp_classifier = MLP_Classifier(in_channels = newD.x.shape[1], hid_channels = 256, 
                                                    out_channels = dataset.y.unique().shape[0], 
                                                    dp = 0.3).to(device)

                    trainer = ClassifierTrainer(graph = newD, train_nodes = cur_train, 
                                                val_nodes = cur_valid, test_nodes = cur_test,
                                                device = device, hidden_dim = dim, model_type = "mlp")


                    mlp_classifier = trainer.fit(newD, model = mlp_classifier, lr = 1e-3, 
                                                 weight_decay = 1e-6, epochs = 500, early_stop = 300)

                    train_acc, valid_acc, test_acc, _ = trainer.eval(newD, mlp_classifier)
                    parformance_table1[i, idid] = valid_acc
                    parformance_table2[i, idid] = test_acc

                else : ## Clustering

                    kmeans = KMeans(n_clusters=dataset.y.unique().shape[0],  
                                    random_state=i, n_init=10).fit(newZ.cpu().numpy())
                    pred_labels = kmeans.labels_

                    valid_acc = nmi_score(gt_labels[cur_valid], pred_labels[cur_valid])
                    test_acc = nmi_score(gt_labels, pred_labels)

                    parformance_table1[i, idid] = valid_acc
                    parformance_table2[i, idid] = test_acc

            print("Seed:", i)
            print("Validation:", parformance_table1[i, :])
            print("Test:", parformance_table2[i, :])
            print()
        
    print("Final performance!")
    print("Average across SSL epochs.")
    print("Validation:", np.mean(parformance_table1, axis = 0))
    print("Test:", np.mean(parformance_table2, axis = 0))
    print("STD across SSL epochs.")
    print("Validation:", np.std(parformance_table1, axis = 0))
    print("Test:", np.std(parformance_table2, axis = 0))
    
