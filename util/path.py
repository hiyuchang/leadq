import os


def set_result_dir(args): 
    
    if "shard" in args.partition:
        args.result_dir = '{}/save/{}/{}_num{}_C{}_r{}_le{}_arrive{}_query{}/{}_{}/seed{}/reset_{}/qmode_{}/{}/label{:06d}/'.format(
            args.save_dir,
            args.dataset,
            args.model,
            args.num_users, 
            args.frac, 
            args.rounds,
            args.local_ep, 
            args.n_arrive,
            args.n_query,
            args.partition,
            args.num_classes_per_user,
            args.seed, 
            args.reset,
            args.query_model_mode,
            args.al_method, 
            args.n_current) # args.current_ratio)         
        
    elif "dir" in args.partition:
        args.result_dir = '{}/save/{}/{}/{}_num{}_C{}_r{}_le{}_arrive{}_query{}/{}_{}/seed{}/reset_{}/qmode_{}/{}/label{:06d}/'.format(
            args.save_dir,
            args.fl_algo,
            args.dataset, 
            args.model,
            args.num_users, 
            args.frac, 
            args.rounds,
            args.local_ep,  
            args.n_arrive,
            args.n_query,
            args.partition,
            args.dd_alpha, 
            args.seed, 
            args.reset,
            args.query_model_mode,
            args.al_method, 
            args.n_current) # args.current_ratio)  
        
    # if args.query_ratio == args.current_ratio:
    if args.n_current == args.n_init * args.num_users:
        args.query_model = None
    else:
        # use last.pt for previous ratio
        # print(args.result_dir[:-7])
        args.query_model = args.result_dir[:-7] + '{:06d}/'.format(args.n_current - args.n_arrive_tot)
        if args.custom_name is not None:
            args.query_model += "{}/".format(args.custom_name)
        args.query_model += "last.pt"
    
    if args.custom_name is not None:
        args.result_dir += "{}/".format(args.custom_name)

    if not os.path.exists(os.path.join(args.result_dir)):
        os.makedirs(os.path.join(args.result_dir), exist_ok=True)
        
    return args


def set_dict_user_path(args):
    
    if "shard" in args.partition:
        args.dict_user_path = "{}/save/dict_users_{}/{}_num{}_C{}_r{}_le{}_arrive{}_query{}/{}_{}/seed{}/".format(
            args.save_dir,
            args.dataset,
            args.model,
            args.num_users, 
            args.frac, 
            args.rounds,
            args.local_ep, 
            args.n_arrive,
            args.n_query,
            args.partition,
            args.num_classes_per_user, 
            args.seed)
        
    elif "dir" in args.partition:
        args.dict_user_path = "{}/save/{}/dict_users_{}/{}_num{}_C{}_r{}_le{}_arrive{}_query{}/{}_{}/seed{}/".format(
            args.save_dir,
            args.fl_algo,
            args.dataset,
            args.model,
            args.num_users, 
            args.frac, 
            args.rounds,
            args.local_ep, 
            args.n_arrive,
            args.n_query, 
            args.partition,
            args.dd_alpha, 
            args.seed)
        
    # Save dict_users for next round
    args.dict_user_path = args.dict_user_path + "reset_{}/qmode_{}/{}".format(args.reset, args.query_model_mode, args.al_method)
    if args.custom_name is not None:
        args.dict_user_path += "/{}".format(args.custom_name)
        
    if not os.path.exists(args.dict_user_path):
        os.makedirs(args.dict_user_path)
        
    return args