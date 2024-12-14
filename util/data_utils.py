import numpy as np


def padding_data(data_idx, batch_size):
    # padding
    data_idx = np.array(data_idx)
    if len(data_idx) < batch_size:
        n_repeat = batch_size // len(data_idx) + (batch_size % len(data_idx) > 0) 
        data_idx = np.repeat(data_idx, n_repeat, axis=0)

    if len(data_idx) % batch_size != 0:
        n_batch = len(data_idx) // batch_size  # Calculate how many full batches can be formed
        data_idx = data_idx[:int(batch_size * n_batch)]  # Keep only the full batches
    return data_idx


def new_sample_arrive(store, n_arrive, fix_seed, recycle):
    """Sample some data from the unlabeled pool and add them to the arrived"""
    pool = store["dict_users_train_unlabel"]
    already_arrived = store["dict_users_train_arrive"]
    arrived = {}
    np.random.seed(fix_seed)
    for i, samples in pool.items():
        if len(samples) < n_arrive:
            if recycle:
                pool[i] = np.union1d(samples, already_arrived[i]) # remained+recycled
                samples = pool[i]
                already_arrived[i] = [] # clear
            else:
                continue
        new = np.random.choice(samples, n_arrive)
        arrived[i] = new
        pool[i] = np.setdiff1d(samples, new) # remove from unlabeled pool
        already_arrived[i] = np.union1d(already_arrived[i], new) # add to arrived
    store["dict_users_train_arrive"] = arrived
    store["dict_users_train_unlabel"] = pool
    store["dict_users_train_hist"] = already_arrived
    return store
    