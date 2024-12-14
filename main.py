import os
import random
import copy
import pickle
import numpy as np
import pandas as pd
import wandb
import torch
from torchvision import datasets, transforms

from models import get_model
from fl_methods import get_fl_method_class
from query_strategies import random_query_samples, algo_query_samples
from util.args import args_parser
from util.path import set_result_dir, set_dict_user_path
from util.data_simulator import shard_balance, dir_balance
from util.misc import adjust_learning_rate
from util import new_sample_arrive


def get_dataset(args):
    MEAN = {
        "svhn": [0.4376821, 0.4437697, 0.47280442],
        "cifar100": [0.507, 0.487, 0.441],
    }
    STD = {
        "svhn": [0.19803012, 0.20101562, 0.19703614],
        "cifar100": [0.267, 0.256, 0.276],
    }

    noaug = [
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN[args.dataset], std=STD[args.dataset]),
    ]

    weakaug = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN[args.dataset], std=STD[args.dataset]),
    ]

    trans_noaug = transforms.Compose(noaug)
    trans_weakaug = transforms.Compose(weakaug)

    # standard benchmarks
    print("Load Dataset {}".format(args.dataset))
    if args.dataset == "svhn":
        dataset_train = datasets.SVHN(
            args.data_dir, "train", download=True, transform=trans_weakaug
        )
        dataset_query = datasets.SVHN(
            args.data_dir, "train", download=True, transform=trans_noaug
        )
        dataset_test = datasets.SVHN(
            args.data_dir, "test", download=True, transform=trans_noaug
        )

    elif args.dataset == "cifar100":
        dataset_train = datasets.CIFAR100(
            args.data_dir, train=True, download=True, transform=trans_weakaug
        )
        dataset_query = datasets.CIFAR100(
            args.data_dir, train=True, download=True, transform=trans_noaug
        )
        dataset_test = datasets.CIFAR100(
            args.data_dir, train=False, download=True, transform=trans_noaug
        )

    else:
        exit("Error: unrecognized dataset")

    # args.dataset_train = dataset_train
    args.total_data = len(dataset_train)

    if args.partition == "shard_balance":
        dict_users_train_total = shard_balance(dataset_query, args)
    elif args.partition == "dir_balance":
        dict_users_train_total, _ = dir_balance(dataset_query, args)
    dict_users_test_total = None

    # keep some test data for evaluation
    total_indices = np.arange(len(dataset_test))
    eval_indices = np.random.choice(total_indices, args.n_eval, replace=False)
    test_indices = np.setdiff1d(total_indices, eval_indices)

    # Create the subsets
    dataset_eval = torch.utils.data.Subset(dataset_test, eval_indices)
    dataset_test = torch.utils.data.Subset(dataset_test, test_indices)

    # for initilization
    if not args.resume_ratio:
        args.query_ratio = round(args.n_query_tot / args.total_data, 4)
        args.current_ratio = round(args.n_init * args.num_users / args.total_data, 4)
    else:
        args.current_ratio = args.resume_ratio
        args.n_current = args.resume_n

    return (
        dataset_train,
        dataset_query,
        dataset_test,
        dataset_eval,
        dict_users_train_total,
        dict_users_test_total,
        args,
    )


def train_test(net_glob, dataset_train, dataset_test, dict_users_train_label, args):
    results_save_path = os.path.join(args.result_dir, "results.csv")

    fl_method = get_fl_method_class(args.fl_algo)(args, dict_users_train_label)

    results = []
    for fl_round in range(args.rounds):
        w_glob = None
        loss_locals = []
        args.g_norms = [None for i in range(args.num_users)]
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        lr = adjust_learning_rate(args, fl_round)

        total_data_num = sum([len(dict_users_train_label[idx]) for idx in idxs_users])

        fl_method.on_round_start(net_glob=net_glob)

        for idx in idxs_users:
            fl_method.on_user_iter_start(dataset_train, idx)

            net_local = copy.deepcopy(net_glob)
            w_local, loss, g_norm = fl_method.train(
                net=net_local.to(args.device),
                user_idx=idx,
                lr=lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
            loss_locals.append(copy.deepcopy(loss))
            args.g_norms[idx] = g_norm

            fl_method.on_user_iter_end()

            w_glob = fl_method.aggregate(
                w_glob=w_glob,
                w_local=w_local,
                idx_user=idx,
                total_data_num=total_data_num,
            )

        fl_method.on_round_end(idxs_users)

        net_glob.load_state_dict(w_glob, strict=False)

        # test model
        if fl_round == 0 or (fl_round + 1) % 10 == 0:
            acc_test, loss_test = fl_method.test(net_glob, dataset_test)

            loss_avg = sum(loss_locals) / len(loss_locals)  # training loss
            print(
                "FL Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}".format(
                    fl_round, loss_avg, loss_test, acc_test
                )
            )
            results.append(np.array([fl_round, loss_avg, loss_test, acc_test]))
            wandb.log(
                {
                    "FL/Round": fl_round + args.cum_round,
                    "Train/Loss": loss_avg,
                    "Test/Acc": acc_test,
                    "Test/Loss": loss_test,
                }
            )

    args.cum_round += args.rounds
    last_save_path = os.path.join(args.result_dir, "last.pt")
    torch.save(net_glob.state_dict(), last_save_path)

    final_results = np.array(results)
    final_results = pd.DataFrame(
        final_results, columns=["epoch", "loss_avg", "loss_test", "acc_test"]
    )
    final_results.to_csv(results_save_path, index=False)
    wandb.log(
        {
            "AL/Round": al_round,
            "N_Arrived": args.n_arrived,
            "N_Queried": args.n_queried,
            "Avg/Loss(F)": round(loss_avg, 2),
            "Test/Acc(F)": round(acc_test, 2),
        }
    )
    print(
        "AL/Round:", al_round, "Arrived samples:", args.n_arrived, "Queried samples: ", args.n_queried, "Test accuracy:", round(acc_test, 2)
    )

    return net_glob.state_dict()


if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device(
        "cuda:{}".format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else "cpu"
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    args = set_result_dir(args)
    args = set_dict_user_path(args)

    # dataset for each client
    (
        dataset_train,
        dataset_query,
        dataset_test,
        dataset_eval,
        dict_users_train_total,
        dict_users_test_total,
        args,
    ) = get_dataset(args)

    store = {
        "dict_users_train_total": dict_users_train_total,
        "dict_users_train_label": {i: [] for i in range(args.num_users)},
        "dict_users_train_unlabel": copy.deepcopy(dict_users_train_total),
        "dict_users_train_hist": {i: [] for i in range(args.num_users)},  # all past arrived
        "dict_users_train_arrive": {i: [] for i in range(args.num_users)},  # arrived in one round
        "dataset_query": dataset_query,
        "dataset_train": dataset_train,
        "dataset_test": dataset_test,
        "dataset_eval": dataset_eval,
    }
    """
    NOTE:
    # dataset_train has weak augmentation 
    # dataset_query is raw data
    # dataset_test only for extracting ten synthetic data
    """

    # wandb initialization
    wandb.init(
        project="federated_al",
        name="{}-{}-A{}-Q{}".format(
            args.al_method, args.fl_algo, args.n_arrive, args.n_query
        ),
        config=args,
    )

    # init environment
    if args.al_method == "leadq":
        from util.env_for_marl import FLEnv

        env = FLEnv(args, store)
        args.env = env
        observation, next_observation = None, None
        reward_list = []

    al_round = -1  # -1 is for random initialization
    if args.query_model:
        last_ckpt = torch.load(args.query_model)
        net_glob = get_model(args)
        net_glob.load_state_dict(last_ckpt)
        args.raw_ckpt = copy.deepcopy(net_glob.state_dict())
        al_round = int(args.result_dir.split("/")[-1][-6:0] // args.n_query_tot)
        print(f"Resume from AL round {al_round}")
        if args.al_method == "leadq":
            env.load_model(args.result_dir + "agent_model/")

    while al_round < args.al_round_max:
        args.al_round = al_round
        print(f"*********** AL round {al_round} ***********")
        print("[Current queried data ratio] %.4f" % args.current_ratio)
        print("[Current arrived data number] %d" % args.n_current)
        net_glob = get_model(args)

        # new samples arrive
        store = new_sample_arrive(
            store, args.n_arrive, al_round + args.seed, recycle=not args.no_recycle
        )

        # get observation
        if al_round >= 0 and args.al_method == "leadq":
            observation = next_observation
            args.observation = observation
            next_observation = None

        # query
        if al_round == -1:
            # random initialization args.n_init samples
            dict_users_train_label, store, args = random_query_samples(
                dict_users_train_total, store, args
            )
            args.cum_round = 0

            # initialize the raw_ckpt
            args.raw_ckpt = copy.deepcopy(net_glob.state_dict())
        else:
            if dict_users_train_label is None:
                path = os.path.join(
                    args.dict_user_path,
                    "dict_users_train_label_{}.pkl".format(
                        args.n_current - args.n_arrive_tot
                    ),
                )
                with open(path, "rb") as f:
                    dict_users_train_label = pickle.load(f)
                args.dict_users_total_path = os.path.join(
                    args.dict_user_path,
                    "dict_users_train_test_total.pkl".format(args.seed),
                )
                store["dict_users_train_label"] = dict_users_train_label
                last_ckpt = torch.load(args.query_model)

            dict_users_train_label, store = algo_query_samples(
                dataset_train, dataset_query, store, args
            )

        if args.reset == "continue" and args.query_model:
            query_net_state_dict = torch.load(args.query_model)
            net_glob.load_state_dict(query_net_state_dict)

        last_ckpt = train_test(
            net_glob, dataset_train, dataset_test, dict_users_train_label, args
        )
        if args.al_method == "leadq":
            if not os.path.exists(args.result_dir + "agent_model/"):
                os.makedirs(args.result_dir + "agent_model/", exist_ok=True)
            env.save_model(args.result_dir + "agent_model/")

        # get next observation
        if args.al_method == "leadq":
            next_observation = env.get_observation(net_glob, store, args)
        
            # get feedback
            if al_round >= 0:
                reward, state = env.step(
                    net_glob
                )  # actions are included in updated net_glob
                reward_list.append(reward)

                # collect information
                env.collect_info(
                    observation=observation,
                    action=args.action,
                    feedback=(reward, state),
                    next_observation=next_observation,
                )
                env.learn()

        args.current_ratio += args.query_ratio
        args.n_current += args.n_arrive_tot
        al_round += 1

        # update path
        args = set_result_dir(args)
        args = set_dict_user_path(args)
