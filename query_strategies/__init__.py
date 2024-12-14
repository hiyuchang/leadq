import os
import copy
import torch
import numpy as np

from models import get_model
from .random_sampling import RandomSampling
from .least_confidence import LeastConfidence
from .core_set import CoreSet
from .fal import LoGo, KAFAL, LeaDQ


def random_query_samples(dict_users_train_total, store, args):
    """ randomly select the labeled samples at the first round
    """
    args.dict_users_total_path = os.path.join(args.dict_user_path, 'dict_users_train_test_total.pkl'.format(args.seed))
    dict_users_train_label = {user_idx: [] for user_idx in dict_users_train_total.keys()}

    # sample n_start example on each client
    for idx in dict_users_train_total.keys():
        dict_users_train_label[idx] = np.random.choice(np.array(list(dict_users_train_total[idx])), args.n_init, replace=False)
        
    store["dict_users_train_label"] = dict_users_train_label
    store["dict_users_train_hist"] = copy.deepcopy(dict_users_train_label)
    args.n_queried = args.n_init * args.num_users
    args.n_arrived = args.n_init * args.num_users
    
    return dict_users_train_label, store, args
    
    
def algo_query_samples(dataset_train, dataset_query, store, args):
    """ query samples from the unlabeled pool
    """
    dict_users_train_total = store["dict_users_train_total"]
    dict_users_train_label = store["dict_users_train_label"]
    dict_users_train_arrive = store["dict_users_train_arrive"]

    # Build model
    query_net = get_model(args)

    query_net_state_dict = torch.load(args.query_model)
    query_net.load_state_dict(query_net_state_dict)    
            
    flag_fal = False
    # AL baselines
    if args.al_method == "random":
        strategy = RandomSampling(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "conf":
        strategy = LeastConfidence(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "coreset":
        strategy = CoreSet(dataset_query, dataset_train, query_net, args)
        
    # FAL baselines
    elif args.al_method == "logo":
        strategy = LoGo(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "kafal":
        strategy = KAFAL(dataset_query, dataset_train, query_net, args)
    
    # proposed method and a variant of coreset
    elif args.al_method == "leadq":
        strategy = LeaDQ(dataset_query, dataset_train, query_net, args)
        flag_fal = True
        
    else:
        exit('There is no such AL methods') 
    
    # global selection
    if flag_fal: 
        unlabel_idxs_tot = []
        for unlabel_idxs in store["dict_users_train_arrive"].values():
            unlabel_idxs_tot.extend(unlabel_idxs)
        
        if args.al_method == "leadq":
            new_data_list = strategy.query(args.observation, store)
        
        else:
            unlabel_idxs_tot = np.array(unlabel_idxs_tot)
            new_data_list = strategy.query(store, unlabel_idxs_tot, n_query=args.n_query_tot)
        
        for user_idx in dict_users_train_total.keys():
            label_idxs = dict_users_train_label[user_idx]
            unlabel_idxs = dict_users_train_arrive[user_idx]
            
            if len(unlabel_idxs) > 0:
                if isinstance(new_data_list, dict):
                    new_data = new_data_list[user_idx]
                else:
                    new_data = np.intersect1d(unlabel_idxs, new_data_list)
                dict_users_train_label[user_idx] = np.array(list(new_data) + list(label_idxs)) 
    else:
        new_data_list = []
        for user_idx in dict_users_train_total.keys(): 
            label_idxs = dict_users_train_label[user_idx]
            unlabel_idxs = dict_users_train_arrive[user_idx]
            
            new_data = strategy.query(user_idx, label_idxs, unlabel_idxs, args.n_query)
            if isinstance(new_data, int) or isinstance(new_data, np.int64):
                new_data = [new_data]
            dict_users_train_label[user_idx] = np.array(list(new_data) + list(label_idxs)) 
            new_data_list.extend(new_data)

    store["dict_users_train_total"] = dict_users_train_total
    store["dict_users_train_label"] = dict_users_train_label
    args.n_queried += args.n_query_tot
    args.n_arrived += args.n_arrive_tot

    return dict_users_train_label, store
    