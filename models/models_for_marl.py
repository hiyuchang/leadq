import torch
import torch.nn as nn
import torch.nn.functional as F

    
# Ref: https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/3.QMIX_VDN_SMAC/mix_net.py
class QMIX_Net(nn.Module):
    def __init__(self, args, state_dim, n_agents):
        super(QMIX_Net, self).__init__()
        self.N = n_agents
        self.state_dim = state_dim
        self.batch_size = args.batch_size_rl
        self.qmix_hidden_dim = args.qmix_hidden_dim
        self.hyper_hidden_dim = args.hyper_hidden_dim
        self.hyper_layers_num = args.hyper_layers_num
        """
        w1:(N, qmix_hidden_dim)
        b1:(1, qmix_hidden_dim)
        w2:(qmix_hidden_dim, 1)
        b2:(1, 1)
        first outputs a vector and then transform it to a matrix (as hyper_w1 needs)
        """
        if self.hyper_layers_num == 2:
            self.hyper_w1 = nn.Sequential(nn.Linear(self.state_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, self.N * self.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(self.state_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, self.qmix_hidden_dim * 1))
        elif self.hyper_layers_num == 1:
            self.hyper_w1 = nn.Linear(self.state_dim, self.N * self.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.state_dim, self.qmix_hidden_dim * 1)
        else:
            raise ValueError("hyper_layers_num should be 1 or 2")

        self.hyper_b1 = nn.Linear(self.state_dim, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, 1))

    def forward(self, q, s):
        # q.shape(batch_size, max_episode_len, N)
        # s.shape(batch_size, max_episode_len,state_dim)
        q = q.view(-1, 1, self.N)  # (batch_size * max_episode_len, 1, N)
        s = s.reshape(-1, self.state_dim)  # (batch_size * max_episode_len, state_dim)

        w1 = torch.abs(self.hyper_w1(s))  # (batch_size * max_episode_len, N * qmix_hidden_dim)
        b1 = self.hyper_b1(s)  # (batch_size * max_episode_len, qmix_hidden_dim)
        w1 = w1.view(-1, self.N, self.qmix_hidden_dim)  # (batch_size * max_episode_len, N,  qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        # torch.bmm: 3 dimensional tensor multiplication
        q_hidden = F.elu(torch.bmm(q, w1) + b1)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        w2 = torch.abs(self.hyper_w2(s))  # (batch_size * max_episode_len, qmix_hidden_dim * 1)
        b2 = self.hyper_b2(s)  # (batch_size * max_episode_len,1)
        w2 = w2.view(-1, self.qmix_hidden_dim, 1)  # (batch_size * max_episode_len, qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)  # (batch_size * max_episode_len, 1， 1)

        q_total = torch.bmm(q_hidden, w2) + b2  # (batch_size * max_episode_len, 1， 1)
        q_total = q_total.view(self.batch_size, -1, 1)  # (batch_size, max_episode_len, 1)
        return q_total

    
# VDN（Value-Decomposition Networks For CooperativeMulti-Agent Learning）
class VDN_Net(nn.Module):
    def __init__(self, ):
        super(VDN_Net, self).__init__()

    def forward(self, q):
        return torch.sum(q, dim=-1, keepdim=True)  # (batch_size, max_episode_len, 1)


# Ref: https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/3.QMIX_VDN_SMAC/qmix_smac.py
class Q_network_RNN(nn.Module):
    def __init__(self, args, input_dim, action_dim):
        super(Q_network_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, action_dim)
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, inputs):
        # When 'choose_action', inputs.shape(N, input_dim)
        # When 'train', inputs.shape(bach_size*N,input_dim)
        x = F.relu(self.fc1(inputs))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        Q = self.fc2(self.rnn_hidden)
        return Q
    
    def init_hidden_state(self):
        self.rnn_hidden = None


# Ref: https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/3.QMIX_VDN_SMAC/qmix_smac.py
class Q_network_MLP(nn.Module):
    def __init__(self, args, input_dim, action_dim):
        super(Q_network_MLP, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, action_dim)
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, inputs):
        # When 'choose_action', inputs.shape(N,input_dim)
        # When 'train', inputs.shape(bach_size,max_episode_len,N,input_dim)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x)
        return Q


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
