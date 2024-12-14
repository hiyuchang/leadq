import random
import numpy as np

from .strategy import Strategy


class RandomSampling(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        if isinstance(unlabel_idxs, np.ndarray):
            unlabel_idxs = unlabel_idxs.tolist()
        return random.sample(unlabel_idxs, n_query)