import os
import torch
import torch.nn as nn
import numpy as np
import copy
import wandb
from torch.utils.data import DataLoader
from torch.autograd import Variable

from query_strategies.strategy import DatasetSplit
from models.models_for_marl import *
from util import padding_data


class FLEnv:
    def __init__(self, args, store):
        self.n_arrive = args.n_arrive
        self.n_eval = args.n_eval
        self.num_classes = args.num_classes
        self.n_query = args.n_query
        self.test_bs = args.test_bs
        self.N = args.num_users
        self.batch_size = args.batch_size_rl
        self.device = args.device
        self.dataset_eval = store["dataset_eval"]
        
        # local observation 
        self.obs_dim = args.num_classes * args.n_arrive
        
        # global state: prediction on evaluation data
        self.state_dim = args.n_eval  # The dimension of global state space
        self.sel_act_dim = args.n_query  # The dimensions of the selected action space (may > 1)
        self.n_action = args.n_arrive # The dimensions of an agent's action space
        self.episode_limit = args.episode_limit
        self.max_train_steps = args.max_train_steps
        
        self.cur_timestamp = -1  # current timestamp (> episode_limit)
        self.acc_val_list = [0.1]
        self.total_steps = 0 # for lr decay
        
        # initialize the agents
        self.agents_n = QLearner(self.get_env_info(), args)
        
        # others
        self.replay_buffer = ReplayBuffer(self.get_env_info())
        
        print("Dimensions are: obs_dim={}, state_dim={}, n_action={}".format(self.obs_dim, self.state_dim, self.n_action))
        
    def get_cur_timestamp(self):
        return self.cur_timestamp
    
    def get_env_info(self):
        env_info = {
            "n_agents": self.N,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim, 
            "sel_act_dim": self.sel_act_dim,
            "n_action": self.n_action,
            "episode_limit": self.episode_limit,
            "batch_size_rl": self.batch_size,
            "device": self.device,
        }
        return env_info
    
    def collect_info(self, observation, action, feedback, next_observation=None):
        self.cur_timestamp += 1
        
        reward, state = feedback
        timestep_info = {
            'obs_n': observation.cpu().tolist(),
            's': state,
            'a_n': action,
            'avail_a_n': [[1 for _ in range(self.N)] for _ in range(self.n_action)],
            'r': reward,
            'dw': 1 if self.cur_timestamp % self.episode_limit == 0 else 0,
            'active': 1,
        }
        
        self.replay_buffer.store_transition(timestep_info)
        return

    def learn(self):
        # update agent policy and mixing network 
        if self.replay_buffer.current_size >= self.batch_size * self.episode_limit:
            step = 0 
            while step < self.max_train_steps:
                # in fact, this is learned by all agents
                self.agents_n.train(self.replay_buffer, self.total_steps)
                step += 1
                self.total_steps += self.episode_limit
            print("The policy and mixing network are updated.")
        return
    
    def make_action(self, observation, dict_arrive, g_net): 
        def _to_data_idx(data_arrive, action):
            return data_arrive[action]  
        
        use_policy = True if self.cur_timestamp > self.batch_size else False
        action = self.agents_n.choose_action(observation, dict_arrive, g_net, use_policy)
        
        query_idx = []
        for i in range(self.N):
            act = action[i]
            if i in dict_arrive: # sometimes no data arrival
                query_idx.append(_to_data_idx(dict_arrive[i], act))
            else:
                print(f"{i} is not in {list(dict_arrive.keys())}")
                query_idx.append([])
        
        return query_idx, action
    
    def get_observation(self, g_net, store, args):
        dict_arrive = store["dict_users_train_arrive"]
        
        observation = []
        for i in range(self.N):
            new_samples = dict_arrive[i]
            obs = self.agents_n.get_local_obs(new_samples, g_net, store, agent_id=i)
            observation.append(obs)
        return torch.stack(observation, dim=0)
    
    def step(self, net): 
        # evaluate current global net on the eval dataset
        data_loader = DataLoader(self.dataset_eval, batch_size=self.test_bs)
        data_nums = len(data_loader.dataset)
        loss_func = nn.CrossEntropyLoss(reduction='none')

        net.eval()
        
        test_loss, correct = 0, 0
        state = []
        with torch.no_grad():
            for idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output, emb = net(data)
                loss = loss_func(output, target)
                test_loss += loss.sum().item()
                
                y_pred = output.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.view_as(y_pred)).detach().sum().item()

                state.extend(loss.detach().cpu().tolist())  # Convert tensor of losses to a list and extend state
            test_loss /= data_nums
            accuracy = 100.00 * float(correct) / data_nums

        # get reward
        reward = accuracy - self.acc_val_list[-1]
        self.acc_val_list.append(accuracy)        
        wandb.log({"Val/Acc": accuracy, "Val/Loss": test_loss}) 
        
        return reward, state
    
    def save_model(self, result_dir):
        self.agents_n.save_model(result_dir)
        return
    
    def load_model(self, model_path):
        self.agents_n.eval_Q_net.load_state_dict(torch.load(model_path))
        self.agents_n.target_Q_net.load_state_dict(torch.load(model_path))
        return
    
    def get_pred_scores(self):
        return self.agents_n.all_pred_scores

class QLearner:
    """Agent for MARL"""
    def __init__(self, env_info, args):
        self.obs_dim = env_info["obs_dim"]
        self.action_dim = env_info["n_action"]
        self.state_dim = env_info["state_dim"]
        self.sel_act_dim = env_info["sel_act_dim"]
        self.n_arrive = args.n_arrive
        self.N = env_info["n_agents"]
        self.device = env_info["device"]
        self.gamma = args.gamma
        self.num_classes = args.num_classes
        self.use_lr_decay = args.use_lr_decay
        self.batch_size = args.batch_size_rl
        self.n_query = args.n_query
        self.max_train_steps = args.max_train_steps
        self.target_update_freq = args.target_update_freq
        self.tau = args.tau
        self.use_hard_update = args.use_hard_update
        self.use_rnn = args.use_rnn
        self.use_qmix = args.use_qmix
        self.use_double_q = args.use_double_q
        self.use_grad_clip = args.use_grad_clip
        self.input_dim = self.obs_dim
        self.all_pred_scores = {i: [100] * args.n_train_tot for i in range(self.N)}  # 100 means will not be selected
        
        # define Q-network
        if self.use_rnn:
            self.eval_Q_net = Q_network_RNN(args, self.input_dim, self.action_dim)
            self.target_Q_net = Q_network_RNN(args, self.input_dim, self.action_dim)
        else:
            self.eval_Q_net = Q_network_MLP(args, self.input_dim, self.action_dim)
            self.target_Q_net = Q_network_MLP(args, self.input_dim, self.action_dim)
        self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
        
        # define mixing network
        if self.use_qmix:
            self.eval_mix_net = QMIX_Net(args, self.state_dim, self.N)
            self.target_mix_net = QMIX_Net(args, self.state_dim, self.N)
        else:
            self.eval_mix_net = VDN_Net()
            self.target_mix_net = VDN_Net()
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        
        # define optimizer
        self.eval_parameters = list(self.eval_mix_net.parameters()) + list(self.eval_Q_net.parameters())
        self.optimizer = torch.optim.Adam(self.eval_Q_net.parameters(), lr=0.01)
        self.train_step = 0
        
    def get_local_obs(self, new_samples, g_net, store, agent_id):
        self.dataset_query = store["dataset_query"]
        self.dataset_train = store["dataset_train"]
        self.g_net = g_net
        
        prediction = self.predict_prob(new_samples, net=g_net)

        for idx, pred in zip(new_samples, prediction):
            self.all_pred_scores[agent_id][idx] = pred.max().item()       
        
        _obs_list = [prediction]
     
        observation = torch.cat(_obs_list, dim=1)  # torch.Size([10, 522])
        observation = observation.flatten()
        return observation
    
    def choose_action(self, observation, dict_arrive, g_net, use_policy):
        def least_conf_query(unlabel_idxs, net, n_query):
            probs = self.predict_prob(unlabel_idxs, net)
            U = probs.max(1)[0]
            return U.sort()[1][:n_query].tolist()
        
        if use_policy:
            # print("Using policy to choose action", end=':')
            if True:
                if self.use_rnn:
                    self.eval_Q_net.rnn_hidden = None
                
                self.eval_Q_net.eval()
                self.eval_Q_net.to(self.device)
                
                with torch.no_grad():
                    observation = observation.to(self.device)
                    Q_value = self.eval_Q_net(observation) # observation.shape=torch.Size([10, 5320])
                    action = torch.topk(Q_value, self.sel_act_dim, dim=1).indices
                    action = action.cpu().tolist()
        else:
            # print("Using least conf query to choose action", end=':')
            action = []
            for i in range(self.N):
                act = least_conf_query(dict_arrive[i], g_net, self.sel_act_dim) 
                action.append(act)
                
        # print(action)
        return action

    def train(self, replay_buffer, total_steps):
        batch, max_episode_len = replay_buffer.sample()  # Get training data
        self.train_step += 1

        inputs = self.get_inputs(batch, max_episode_len)  # inputs.shape=(bach_size,max_episode_len+1,N,input_dim)
        actions = batch['a_n'].to(self.device) # actions.shape=(batch_size,max_episode_len,N, sel_act_dim=n_query)
        states = batch['s'].to(self.device) # states.shape=(batch_size,max_episode_len+1,state_dim)
        rewards = batch['r'].unsqueeze(-1).to(self.device) # rewards.shape=(batch_size,max_episode_len,1)
        dones = batch['dw'].unsqueeze(-1).to(self.device) # dones.shape=(batch_size,max_episode_len,1)
        actives = batch['active'].unsqueeze(-1).to(self.device) # actives.shape=(batch_size,max_episode_len,1)
        
        self.eval_Q_net.to(self.device)
        self.target_Q_net.to(self.device)
        
        if self.use_rnn:
            self.eval_Q_net.rnn_hidden = None
            self.target_Q_net.rnn_hidden = None
            q_evals, q_targets = [], []
            for t in range(max_episode_len):  # t=0,1,2,...(episode_len-1)
                q_eval = self.eval_Q_net(inputs[:, t].reshape(-1, self.input_dim))  # q_eval.shape=(batch_size*N,action_dim)
                q_target = self.target_Q_net(inputs[:, t + 1].reshape(-1, self.input_dim))
                q_evals.append(q_eval.reshape(self.batch_size, self.N, -1))  # q_eval.shape=(batch_size,N,action_dim)
                q_targets.append(q_target.reshape(self.batch_size, self.N, -1))

            # Stack them according to the time (dim=1)
            q_evals = torch.stack(q_evals, dim=1)  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
            q_targets = torch.stack(q_targets, dim=1)
        else:
            q_evals = self.eval_Q_net(inputs[:, :-1])  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
            q_targets = self.target_Q_net(inputs[:, 1:])
        
        with torch.no_grad():
            if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions,and use target_net to compute q_target
                q_eval_last = self.eval_Q_net(inputs[:, -1].reshape(-1, self.input_dim)).reshape(self.batch_size, 1, self.N, -1)
                q_evals_next = torch.cat([q_evals[:, 1:], q_eval_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,N,action_dim)
                q_evals_next[batch['avail_a_n'][:, 1:] == 0] = -999999
                # a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, N, 1)
                a_argmax = torch.topk(q_evals_next, self.sel_act_dim, dim=-1).indices
                q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len, N)
            else:
                q_targets[batch['avail_a_n'][:, 1:] == 0] = -999999
                q_targets = q_targets.max(dim=-1)[0]  # q_targets.shape=(batch_size, max_episode_len, N)

        # actions.shape(batch_size,max_episode_len, N, 1)
        q_evals = torch.gather(q_evals, dim=-1, index=actions).squeeze(-1)  # q_evals.shape(batch_size, max_episode_len, N, sel_act_dim)
        
        self.eval_mix_net.to(self.device)
        self.target_mix_net.to(self.device)
        
        _states = states[:, :-1] # exclude the last point
        if self.sel_act_dim > 1:
            q_evals = torch.sum(q_evals, dim=-1, keepdim=True)  # q_evals.shape=(batch_size,max_episode_len,N,1)
            q_targets = torch.sum(q_targets, dim=-1, keepdim=True)
        
        # Compute q_total using QMIX or VDN, q_total.shape=(batch_size, max_episode_len, 1)
        if self.use_qmix:
            q_total_eval = self.eval_mix_net(q_evals, _states)
            q_total_target = self.target_mix_net(q_targets, _states)
        else:
            q_total_eval = self.eval_mix_net(q_evals)
            q_total_target = self.target_mix_net(q_targets)
        
        # targets.shape=(batch_size,max_episode_len,1,sel_act_dim)
        targets = rewards + self.gamma * (1 - dones) * q_total_target

        td_error = (q_total_eval - targets.detach())
        mask_td_error = td_error * actives
        loss = (mask_td_error ** 2).sum() / batch['active'].sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, 10)
        self.optimizer.step()

        if self.use_hard_update:
            # hard update
            if self.train_step % self.target_update_freq == 0:
                self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
                self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        else:
            # Softly update the target networks
            for param, target_param in zip(self.eval_Q_net.parameters(), self.target_Q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.eval_mix_net.parameters(), self.target_mix_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def get_inputs(self, batch, max_episode_len):
        inputs = []
        inputs.append(batch['obs_n'])

        inputs = torch.cat([x for x in inputs], dim=-1)
        return inputs.to(self.device)

    def lr_decay(self, total_steps):  # Learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def save_model(self, result_dir):
        save_path = os.path.join(result_dir, '.pt')
        torch.save(self.eval_Q_net.state_dict(), save_path)
    
    # copy from Class strategy
    def predict_prob(self, unlabel_idxs, net):
        loader_te = DataLoader(DatasetSplit(self.dataset_query, unlabel_idxs), shuffle=False)
        net.eval()
        probs = torch.zeros([len(unlabel_idxs), self.num_classes])
        with torch.no_grad():
            for x, _, idxs in loader_te:
                x = Variable(x.to(self.device))
                output, _ = net(x)
                probs[idxs] = torch.nn.functional.softmax(output, dim=1).cpu().data
        return probs


class ReplayBuffer:
    def __init__(self, env_info):
        self.N = env_info["n_agents"]
        self.obs_dim = env_info["obs_dim"]
        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["n_action"]
        self.episode_limit = env_info["episode_limit"]
        self.batch_size = env_info["batch_size_rl"]
        self.buffer_size = 1000
        self.episode_reward = 0

        self.buffer = {'obs_n': [],
                       's': [],
                       'avail_a_n': [],
                       'a_n': [],
                       'r': [],
                       'dw': [],
                       'active': []}
        self.episode_len = []
        
        self.episode_num = 0
        self.episode_idx = 0
        self.current_size = 0
    
    def store_transition(self, timestep_info):
        episode_idx = self.episode_idx
        # Update the length of the current episode
        if len(self.episode_len) <= self.episode_num:
            self.episode_len.append(episode_idx + 1)
        else:
            self.episode_len[self.episode_num] = episode_idx + 1
        
        # Append transition info directly to the buffer lists
        if episode_idx == 0:  # Start of a new episode, initialize lists
            for key in self.buffer.keys():
                if len(self.buffer[key]) <= self.episode_num:
                    self.buffer[key].append([timestep_info[key]])
                else:
                    self.buffer[key][self.episode_num] = [timestep_info[key]]
        else:  # Existing episode, append the information
            for key in self.buffer.keys():
                self.buffer[key][self.episode_num].append(timestep_info[key])
            self.episode_reward += timestep_info['r']
        
        # Handle index updates and buffer size management
        self.current_size = min(self.current_size + 1, self.buffer_size)
        self.episode_idx += 1
        if self.episode_idx == self.episode_limit - 1:  # End of an episode or terminal state reached
            wandb.log({"Episode": self.episode_num, "Reward/Episode": self.episode_reward})
            self.episode_idx = 0
            self.episode_num += 1 
            self.episode_reward = 0
            
            if self.episode_num >= self.buffer_size:  # Reset to the start if the buffer is full
                self.episode_num = 0
    
    def sample(self):
        # Randomly sample episode indices from the buffer
        indices = np.random.choice(min(self.current_size, len(self.episode_len)), size=self.batch_size, replace=False)
        max_episode_len = int(max([self.episode_len[idx] for idx in indices]))
        
        # Initialize the batch dictionary with lists for each key
        batch = {key: [] for key in self.buffer.keys()}
        
        for key in self.buffer.keys():
            _dtype = torch.long if key == 'a_n' else torch.float32
            
            for idx in indices:
                episode_data = self.buffer[key][idx]
                episode_data_tensor = torch.tensor(np.array(episode_data), dtype=_dtype)
                
                # add padding
                pad_len = max_episode_len + 1 - len(episode_data) if key in ['obs_n', 's', 'avail_a_n', 'last_onehot_a_n'] else max_episode_len - len(episode_data)
                if pad_len > 0:
                    pad_shape = list(episode_data_tensor.shape)
                    pad_shape[0] = pad_len  # Update the padding length
                    pad_value = 1 if key == 'avail_a_n' else 0
                    padding = torch.full(pad_shape, pad_value, dtype=_dtype)
                    episode_data_tensor = torch.cat([episode_data_tensor, padding], dim=0)
                
                batch[key].append(episode_data_tensor)

        for key in batch.keys():
            batch[key] = torch.stack(batch[key], dim=0)
        
        return batch, max_episode_len
