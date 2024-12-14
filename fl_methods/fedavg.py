# Ref: https://github.com/raymin0223/LoGo
import torch
from copy import deepcopy

from .base import FederatedLearning


class FedAvg(FederatedLearning):
    def __init__(self, args, dict_users_train_label=None):
        super().__init__(args, dict_users_train_label)

    def train(self, net, user_idx=None, lr=0.01, momentum=0.9, weight_decay=0.00001):
        net_old = deepcopy(net)
        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        epoch_loss = []  
        for _ in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.data_loader:
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                                
                optimizer.zero_grad()
                output, emb = net(images)
                
                if output.shape[0] == 1:
                    labels = labels.reshape(1,)

                loss = self.loss_func(output, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        grad_norm = 0
        for x, y in zip(net_old.parameters(), net.parameters()):
            grad_norm += torch.norm(x-y, p=2) / lr
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), grad_norm

    