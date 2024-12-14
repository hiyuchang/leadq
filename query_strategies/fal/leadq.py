from ..strategy import Strategy
from util.env_for_marl import *


class LeaDQ(Strategy):
    def __init__(self, dataset_query, dataset_train, net, args):
        super().__init__(dataset_query, dataset_train, net, args)
        
        # initialize the environment        
        self.env = args.env
        self.env_info = self.env.get_env_info()
        self.n_agents = self.env_info["n_agents"]
        self.obs_dim = self.env_info["obs_dim"]  # The dimensions of an agent's observation space
        self.state_dim = self.env_info["state_dim"]  # The dimensions of global state space
        self.action_dim = self.env_info["n_action"]  # The dimensions of an agent's action space
        self.sel_act_dim = self.env_info["sel_act_dim"]
        self.episode_limit = self.env_info["episode_limit"] 
        self.batch_size_rl = self.env_info["batch_size_rl"]
        self.args = args
        
    def query(self, observation, store):
        g_net = self.net
    
        '''make action for query'''
        dict_arrive = store["dict_users_train_arrive"]
        query_idx, action = self.env.make_action(observation, dict_arrive, g_net)
        self.args.action = action

        # get scores
        self.all_pred_scores = self.env.get_pred_scores()
        
        return query_idx


