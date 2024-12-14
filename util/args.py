import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    
    # basic arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--custom_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--save_dir', type=str, default='.', help='when to start saving models')    
    
    # dataset arguments
    parser.add_argument('--data_dir', type=str, default='../datasets/', help='data path')
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="total classes")
    parser.add_argument('--partition', type=str, default="dir_balance", help="methods for Non-IID")
    parser.add_argument('--dd_alpha', type=float, default=0.5, help="concentration parameter for dirichlet distribution")
    parser.add_argument('--num_classes_per_user', type=int, default=2, help="classes per user")
    
    # model arguments
    parser.add_argument('--model', type=str, default='cnn4conv', help='model name')
    parser.add_argument('--pretrained', default=False, action='store_true', help='use pretrained model')
    
    # active learning / data querying arguments
    parser.add_argument('--n_init', type=int, default=None, help="Initital samples")
    parser.add_argument('--al_round_max', type=int, default=500, help="rounds of training")
    parser.add_argument('--resume_ratio', type=float, default=0., help="ratio of data examples for resume")
    parser.add_argument('--query_ratio', type=float, default=0.05, help="ratio of data examples per one query")
    parser.add_argument('--end_ratio', type=float, default=0.0, help="ratio for stopping query")
    parser.add_argument('--query_model_mode', type=str, default="global")
    parser.add_argument('--al_method', type=str, default=None)
    parser.add_argument('--lamb', type=float, default=0.1)
    parser.add_argument('--n_eval', type=int, default=1000, help="eval data samples on server")
    parser.add_argument('--n_arrive', type=int, default=10, help="arrived sample on each client")
    parser.add_argument('--n_query', type=int, default=1, help="number of queried samples on each client")
    parser.add_argument('--fe_type', type=str, default='resnet18', help="feature extraction model")
    parser.add_argument('--no_recycle', default=False, action='store_true')

    # federated learning arguments
    parser.add_argument('--rounds', type=int, default=30, help="rounds of FL training in each querying round")
    parser.add_argument('--num_users', type=int, default=10, help="number of users")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size")
    parser.add_argument('--test_bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--weight_decay', type=float, default=0.00001, help="weight decay (default: 0.00001)")
    parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay ratio")
    parser.add_argument('--reset', type=str, default='random_init', help='doing FL with queried dataset or not')
    parser.add_argument('--fl_algo', type=str, default='fedavg', help='federated learning algorithm')
    parser.add_argument('--mu', type=float, default=0.01, help='weight of FedProx regularization term')

    # multi-agent RL arguments (for LeaDQ)
    parser.add_argument("--episode_limit", type=int, default=10, help="Maximum number of steps per episode")
    parser.add_argument('--gamma', type=float, default=0.99, help="discount factor")
    parser.add_argument('--batch_size_rl', type=int, default=32, help="sampled batch size from buffer")
    parser.add_argument('--use_qmix', type=bool, default=True, help="use QMIX for Q-network")
    parser.add_argument('--use_rnn', type=bool, default=True, help="use RNN for Q-network")
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help="hidden dimension of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument('--use_orthogonal_init', type=bool, default=True, help="use orthogonal initialization for Q-network")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
    parser.add_argument("--max_train_steps", type=int, default=100, help="max train steps")
    parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--hyper_layers_num", type=int, default=1, help="The number of layers of hyper-network")
    parser.add_argument("--use_double_q", type=bool, default=True, help="Whether to use double q-learning")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Whether to use reward normalization")
    parser.add_argument("--use_hard_update", type=bool, default=True, help="Whether to use hard update")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Whether to use gradient clip")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
    parser.add_argument("--tau", type=int, default=0.005, help="If use soft update")

    args = parser.parse_args()
    
    # benchmarks
    if args.dataset == 'svhn':
        args.num_classes = 10
        args.in_channels = 3
        args.img_size = 32
        args.pic_dim = 32 * 32 * 3
        if not args.end_ratio: args.end_ratio = 0.6
    
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.in_channels = 3
        args.img_size = 32
        args.pic_dim = 32 * 32 * 3
        if not args.end_ratio: args.end_ratio = 0.8
    
    if args.n_init is None:
        args.n_init = args.n_arrive
    args.n_query_tot = int(args.n_query * args.num_users)
    args.n_arrive_tot = int(args.n_arrive * args.num_users)

    # for init
    if not args.resume_ratio:
        args.n_current = args.n_init * args.num_users
        
    return args