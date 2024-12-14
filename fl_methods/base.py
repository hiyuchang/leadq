# Ref: https://github.com/raymin0223/LoGo
import copy
import torch.nn as nn
from torch.utils.data import DataLoader

from util.misc import DatasetSplit
from util import padding_data


class FederatedLearning:
    def __init__(self, args, dict_users_train_label=None):
        self.args = args
        self.dict_users_train_label = dict_users_train_label
        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        pass  

    def aggregate(self, w_glob, w_local, idx_user, total_data_num):
        if w_glob is None:
            w_glob = copy.deepcopy(w_local)
            for k in w_glob.keys():
                w_glob[k] = w_local[k] * len(self.dict_users_train_label[idx_user]) / total_data_num
        else:
            for k in w_glob.keys():
                w_glob[k] += w_local[k] * len(self.dict_users_train_label[idx_user]) / total_data_num

        return w_glob

    def test(self, net_g, dataset):
        data_loader = DataLoader(dataset, batch_size=self.args.test_bs)
        data_nums = len(data_loader.dataset)

        net_g.eval()
        
        test_loss, correct = 0, 0
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != -1:
                data, target = data.to(self.args.device), target.to(self.args.device)
            output, _ = net_g(data)

            # sum up batch loss
            test_loss += self.loss_func(output, target).item() * data.size(0)
            # get the index of the max log-probability
            y_pred = output.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= data_nums
        accuracy = 100.00 * float(correct) / data_nums

        return accuracy, test_loss

    def on_round_start(self, net_glob=None):
        pass

    def on_user_iter_start(self, dataset, user_idx):
        data_idx = self.dict_users_train_label[user_idx].copy() # NOTE: copy() is necessary to avoid modifying the original list
        local_bs = self.args.local_bs

        data_idx = padding_data(data_idx, local_bs)
        self.data_loader = DataLoader(DatasetSplit(dataset, data_idx), batch_size=local_bs, shuffle=True, drop_last=True)
        # if not drop_last, the last batch will lead to error (batch norm layer)

    def on_round_end(self, idxs_users=None):
        pass

    def on_user_iter_end(self):
        pass