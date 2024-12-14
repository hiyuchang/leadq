import numpy as np
import torch

from ..strategy import Strategy


class KAFAL(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)

        g_net = self.net
        l_net = self.training_local_only(label_idxs)
        
        g_probs = self.predict_prob(unlabel_idxs, g_net)
        l_probs = self.predict_prob(unlabel_idxs, l_net)
        
        # JS-divergence
        div = (g_probs * torch.log(g_probs / l_probs)).sum(1) \
                + (l_probs * torch.log(l_probs / g_probs)).sum(1)
        
        query_idx = div.argsort(descending=True)[:n_query]
        
        return unlabel_idxs[query_idx]