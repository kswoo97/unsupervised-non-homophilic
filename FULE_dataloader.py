import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, WikiCS, WebKB, WikipediaNetwork, Actor
from torch_geometric.utils import remove_self_loops


def prepare_dataset_and_hyperparameter_config_medium (d_name, downstream, device) : 
    
    if d_name not in ["Cora", "Citeseer", "Pubmed", "Photo", "Computers", "Chameleon", "Squirrel", "Actor", "Cornell", "Wisconsin", "Texas"] : 
        raise TypeError("Check data name")

    if d_name == "Cora" : 
        
        dataset = Planetoid(root='./datasets/{0}'.format(d_name), name=d_name)[0].to(device)
        
        ## This dataset has a single split.
        train_idxs = [list(torch.where(dataset.train_mask)[0].cpu().numpy())]
        valid_idxs = [list(torch.where(dataset.val_mask)[0].cpu().numpy())]
        test_idxs = [list(torch.where(dataset.test_mask)[0].cpu().numpy())]
        
        if downstream == 'classification' : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 2e-3
            wd2 = 1e-6
            K = 10
            dp = 0.1
            tau = 2.0
            do_detach = True
            do_norm = False
            
        else : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 2e-3
            wd2 = 1e-4
            K = 10
            dp = 0.1
            tau = 1.0
            do_detach = False
            do_norm = False
            
    elif d_name == "Citeseer" : 
        
        dataset = Planetoid(root='./datasets/{0}'.format(d_name), name=d_name)[0].to(device)
        isoVs = set(range(dataset.x.shape[0])) - set(dataset.edge_index.unique().cpu().numpy())
        newE = torch.LongTensor([list(isoVs), list(isoVs)]).to(device)
        dataset.edge_index = torch.hstack([dataset.edge_index, newE])
        
        ## This dataset has a single split.
        train_idxs = [list(torch.where(dataset.train_mask)[0].cpu().numpy())]
        valid_idxs = [list(torch.where(dataset.val_mask)[0].cpu().numpy())]
        test_idxs = [list(torch.where(dataset.test_mask)[0].cpu().numpy())]
        
        if downstream == 'classification' : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-4
            wd2 = 1e-1
            K = 5
            dp = 0.5
            tau = 3.0
            do_detach = False
            do_norm = False

        else : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-3
            wd2 = 1e-4
            K = 5
            dp = 0.3
            tau = 0.5
            do_detach = False
            do_norm = False
        
    elif d_name == 'Pubmed' : 
        dataset = Planetoid(root='./datasets/{0}'.format(d_name), name=d_name)[0].to(device)
        transform = T.Compose([T.RemoveIsolatedNodes(), T.NormalizeFeatures()]) #
        dataset = transform(dataset)
        
        ## This dataset has a single split.
        train_idxs = [list(torch.where(dataset.train_mask)[0].cpu().numpy())]
        valid_idxs = [list(torch.where(dataset.val_mask)[0].cpu().numpy())]
        test_idxs = [list(torch.where(dataset.test_mask)[0].cpu().numpy())]
        
        if downstream == 'classification' : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-4
            wd2 = 1e-2
            K = 10
            dp = 0.3
            tau = 0.1
            do_detach = False
            do_norm = False

        else : 
            lr1 = 1e-2
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-4
            wd2 = 1e-2
            K = 5
            dp = 0.5
            tau = 3.0
            do_detach = False
            do_norm = False
            
    elif d_name == "Photo" : 
        dataset = Amazon(root = "./datasets/{0}".format("Photo"), name = "Photo")[0].to(device)
        dataset.edge_index, _ = remove_self_loops(dataset.edge_index)
        isoVs = set(range(dataset.x.shape[0])) - set(dataset.edge_index.unique().cpu().numpy())
        newE = torch.LongTensor([list(isoVs), list(isoVs)]).to(device)
        dataset.edge_index = torch.hstack([dataset.edge_index, newE])
        
        total_nodes = np.arange(dataset.x.shape[0])
        
        train_idxs = []
        valid_idxs = []
        test_idxs = []
        n10 = int(total_nodes.shape[0] * 0.1)
        
        for i in range(10) : 
            np.random.seed(i)
            np.random.shuffle(total_nodes)
            train_idxs.append(list(total_nodes)[:n10])
            valid_idxs.append(list(total_nodes)[n10:n10 + n10])
            test_idxs.append(list(total_nodes)[n10 + n10:])
        
        if downstream == 'classification' : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-4
            wd2 = 1e-1
            K = 10
            dp = 0.5
            tau = 0.1
            do_detach = False
            do_norm = False

        else : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-4
            wd2 = 0.0
            K = 10
            dp = 0.1
            tau = 0.1
            do_detach = False
            do_norm = True
        
    elif d_name in ["Computers"] : 
        dataset = Amazon(root = "./datasets/{0}".format("Computers"), name = "Computers")[0].to(device)
        dataset.edge_index, _ = remove_self_loops(dataset.edge_index)
        isoVs = set(range(dataset.x.shape[0])) - set(dataset.edge_index.unique().cpu().numpy())
        newE = torch.LongTensor([list(isoVs), list(isoVs)]).to(device)
        dataset.edge_index = torch.hstack([dataset.edge_index, newE])
        
        total_nodes = np.arange(dataset.x.shape[0])
        
        train_idxs = []
        valid_idxs = []
        test_idxs = []
        n10 = int(total_nodes.shape[0] * 0.1)
        
        for i in range(10) : 
            np.random.seed(i)
            np.random.shuffle(total_nodes)
            train_idxs.append(list(total_nodes)[:n10])
            valid_idxs.append(list(total_nodes)[n10:n10 + n10])
            test_idxs.append(list(total_nodes)[n10 + n10:])
        
        if downstream == 'classification' : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-4
            wd2 = 1e-1
            K = 5
            dp = 0.3
            tau = 0.1
            do_detach = False
            do_norm = False

        else : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-4
            wd2 = 0.0
            K = 5
            dp = 0.3
            tau = 0.1
            do_detach = False
            do_norm = True


    elif d_name in ["Chameleon"] : 
        dataset = WikipediaNetwork(root='./datasets/{0}'.format(d_name), name=d_name)[0].to(device)
        dataset.edge_index, _ = remove_self_loops(dataset.edge_index)
        transform = T.Compose([T.ToUndirected()])
        dataset = transform(dataset)
        
        train_idxs = []
        valid_idxs = []
        test_idxs = []
        
        ## This dataset has multiple splits.
        for i in range(10) : 
            
            train_idxs.append(list(torch.where(dataset.train_mask[:, i])[0].cpu().numpy()))
            valid_idxs.append(list(torch.where(dataset.val_mask[:, i])[0].cpu().numpy()))
            test_idxs.append(list(torch.where(dataset.test_mask[:, i])[0].cpu().numpy()))
        
        if downstream == 'classification' : 
            lr1 = 1e-2
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-3
            wd2 = 0.0
            K = 10
            dp = 0.3
            tau = 2.0
            do_detach = False
            do_norm = False

        else : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-4
            wd2 = 0.1
            K = 15
            dp = 0.5
            tau = 3.0
            do_detach = False
            do_norm = False
        
    elif d_name in ["Squirrel"] : 
        dataset = WikipediaNetwork(root='./datasets/{0}'.format(d_name), name=d_name)[0].to(device)
        dataset.edge_index, _ = remove_self_loops(dataset.edge_index)
        transform = T.Compose([T.ToUndirected()])
        dataset = transform(dataset)
        
        train_idxs = []
        valid_idxs = []
        test_idxs = []
        
        ## This dataset has multiple splits.
        for i in range(10) : 
            
            train_idxs.append(list(torch.where(dataset.train_mask[:, i])[0].cpu().numpy()))
            valid_idxs.append(list(torch.where(dataset.val_mask[:, i])[0].cpu().numpy()))
            test_idxs.append(list(torch.where(dataset.test_mask[:, i])[0].cpu().numpy()))
        
        if downstream == 'classification' : 
            lr1 = 1e-2
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-3
            wd2 = 0.0
            K = 5
            dp = 0.1
            tau = 0.1
            do_detach = False
            do_norm = False

        else : 
            lr1 = 1e-2
            wd1 = 1e-3
            lamda = 1.0
            epochs1 = 5000

            lr2 = 1e-4
            wd2 = 1e-2
            K = 20
            dp = 0.5
            tau = 3.0
            do_detach = False
            do_norm = False

    elif d_name in ["Cornell"] : 
        dataset = WebKB(root='./datasets/{0}'.format(d_name), name=d_name)[0].to(device)
        dataset.edge_index, _ = remove_self_loops(dataset.edge_index)
        transform = T.Compose([T.ToUndirected(), T.NormalizeFeatures()]) # , 
        dataset = transform(dataset)
        
        train_idxs = []
        valid_idxs = []
        test_idxs = []
        
        ## This dataset has multiple splits.
        for i in range(10) : 
            
            train_idxs.append(list(torch.where(dataset.train_mask[:, i])[0].cpu().numpy()))
            valid_idxs.append(list(torch.where(dataset.val_mask[:, i])[0].cpu().numpy()))
            test_idxs.append(list(torch.where(dataset.test_mask[:, i])[0].cpu().numpy()))
        
        if downstream == 'classification' : 
            lr1 = 1e-2
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 5e-4
            wd2 = 1e-4
            K = 10
            dp = 0.1
            tau = 3.0
            do_detach = False
            do_norm = False

        else : 
            lr1 = 1e-2
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-3
            wd2 = 1e-2
            K = 10
            dp = 0.1
            tau = 0.1
            do_detach = False
            do_norm = True
            
    elif d_name in ["Wisconsin"] : 
        dataset = WebKB(root='./datasets/{0}'.format(d_name), name=d_name)[0].to(device)
        dataset.edge_index, _ = remove_self_loops(dataset.edge_index)
        transform = T.Compose([T.ToUndirected(), T.NormalizeFeatures()]) # , 
        dataset = transform(dataset)
        
        train_idxs = []
        valid_idxs = []
        test_idxs = []
        
        ## This dataset has multiple splits.
        for i in range(10) : 
            
            train_idxs.append(list(torch.where(dataset.train_mask[:, i])[0].cpu().numpy()))
            valid_idxs.append(list(torch.where(dataset.val_mask[:, i])[0].cpu().numpy()))
            test_idxs.append(list(torch.where(dataset.test_mask[:, i])[0].cpu().numpy()))
        
        if downstream == 'classification' : 
            lr1 = 1e-2
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 5e-4
            wd2 = 1e-4
            K = 5
            dp = 0.3
            tau = 3.0
            do_detach = False
            do_norm = False

        else : 
            lr1 = 1e-2
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-4
            wd2 = 1e-1
            K = 20
            dp = 0.3
            tau = 2.0
            do_detach = False
            do_norm = True
            
    elif d_name in ["Texas"] : 
        dataset = WebKB(root='./datasets/{0}'.format(d_name), name=d_name)[0].to(device)
        dataset.edge_index, _ = remove_self_loops(dataset.edge_index)
        transform = T.Compose([T.ToUndirected(), T.NormalizeFeatures()]) # , 
        dataset = transform(dataset)
        
        train_idxs = []
        valid_idxs = []
        test_idxs = []
        
        ## This dataset has multiple splits.
        for i in range(10) : 
            
            train_idxs.append(list(torch.where(dataset.train_mask[:, i])[0].cpu().numpy()))
            valid_idxs.append(list(torch.where(dataset.val_mask[:, i])[0].cpu().numpy()))
            test_idxs.append(list(torch.where(dataset.test_mask[:, i])[0].cpu().numpy()))
        
        if downstream == 'classification' : 
            lr1 = 1e-2
            wd1 = 5e-5
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-4
            wd2 = 1e-2
            K = 5
            dp = 0.3
            tau = 3.0
            do_detach = False
            do_norm = False

        else : 
            lr1 = 1e-2
            wd1 = 5e-5
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-4
            wd2 = 1e-1
            K = 5
            dp = 0.5
            tau = 0.1
            do_detach = False
            do_norm = True
        


    elif d_name in ["Actor"] : 
        dataset = Actor(root='./datasets/{0}'.format(d_name))[0].to(device)
        dataset.edge_index, _ = remove_self_loops(dataset.edge_index)
        transform = T.Compose([T.ToUndirected(), T.NormalizeFeatures()]) # , 
        dataset = transform(dataset)
        
        train_idxs = []
        valid_idxs = []
        test_idxs = []
        
        ## This dataset has multiple splits.
        for i in range(10) : 
            
            train_idxs.append(list(torch.where(dataset.train_mask[:, i])[0].cpu().numpy()))
            valid_idxs.append(list(torch.where(dataset.val_mask[:, i])[0].cpu().numpy()))
            test_idxs.append(list(torch.where(dataset.test_mask[:, i])[0].cpu().numpy()))
        
        if downstream == 'classification' : 
            lr1 = 1e-2
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-3
            wd2 = 0.0
            K = 10
            dp = 0.3
            tau = 3.0
            do_detach = False
            do_norm = False

        else : 
            lr1 = 1e-2
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-3
            wd2 = 1e-2
            K = 10
            dp = 0.5
            tau = 3.0
            do_detach = False
            do_norm = False
            
    return dataset, lr1, wd1, lamda, epochs1, lr2, wd2, K, dp, tau, do_detach, do_norm, train_idxs, valid_idxs, test_idxs


def prepare_dataset_and_hyperparameter_config_large (d_name, downstream, device) : 
    
    try : 
        dataset = torch.load("./datasets/{0}.pt".format(d_name)).to(device)
        data_x = torch.load("./datasets/{0}_pre.pt".format(d_name))
        
    except : 
        raise TypeError("Pre-computed (X, AX, A^2X) should be loaded in ./datasets folder. Refer to the dropbox link in the github.")
        
    if d_name == "Arxiv" : 
        
        total_nodes = np.arange(dataset.x.shape[0])
        
        train_idxs = []
        valid_idxs = []
        test_idxs = []
        n54 = int(total_nodes.shape[0] * 0.54)
        n18 = int(total_nodes.shape[0] * 0.18)

        for i in range(10) : 
            np.random.seed(i)
            np.random.shuffle(total_nodes)
            train_idxs.append(list(total_nodes)[:n54])
            valid_idxs.append(list(total_nodes)[n54:n54 + n18])
            test_idxs.append(list(total_nodes)[n54 + n18:])
        
        if downstream == 'classification' : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.0
            epochs1 = 1000

            lr2 = 1e-3
            wd2 = 1e-6
            K = 5
            dp = 0.1
            tau = 2.0
            do_detach = False
            do_norm = False

        else : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.0
            epochs1 = 1000

            lr2 = 5e-4
            wd2 = 1e-6
            K = 5
            dp = 0.1
            tau = 0.1
            do_detach = False
            do_norm = False
        
    elif d_name == "Flickr" : 
        
        ## This dataset has a single split.
        train_idxs = [list(torch.where(dataset.train_mask)[0].cpu().numpy())]
        valid_idxs = [list(torch.where(dataset.val_mask)[0].cpu().numpy())]
        test_idxs = [list(torch.where(dataset.test_mask)[0].cpu().numpy())]
        
        if downstream == 'classification' : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-3
            wd2 = 1e-6
            K = 5
            dp = 0.1
            tau = 1.0
            do_detach = False
            do_norm = False

        else : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.01
            epochs1 = 1000

            lr2 = 1e-3
            wd2 = 1e-6
            K = 5
            dp = 0.3
            tau = 0.1
            do_detach = False
            do_norm = False
        
    elif d_name == "Penn94" : 
        
        train_idxs = []
        valid_idxs = []
        test_idxs = []
        
        for i in range(10) : 
            
            train_idxs.append(list(torch.where(dataset.train_mask[:, int(i % 5)])[0].cpu().numpy()))
            valid_idxs.append(list(torch.where(dataset.val_mask[:, int(i % 5)])[0].cpu().numpy()))
            test_idxs.append(list(torch.where(dataset.test_mask[:, int(i % 5)])[0].cpu().numpy()))
        
        if downstream == 'classification' : 
            lr1 = 1e-4
            wd1 = 1e-4
            lamda = 0.0
            epochs1 = 1000

            lr2 = 5e-4
            wd2 = 1e-6
            K = 10
            dp = 0.5
            tau = 1.0
            do_detach = False
            do_norm = False

        else : 
            lr1 = 1e-2
            wd1 = 1e-4
            lamda = 0.0
            epochs1 = 1000

            lr2 = 1e-4
            wd2 = 1e-6
            K = 10
            dp = 0.5
            tau = 0.1
            do_detach = False
            do_norm = False
    
    return dataset, data_x, lr1, wd1, lamda, epochs1, lr2, wd2, K, dp, tau, do_detach, do_norm, train_idxs, valid_idxs, test_idxs
    
    
