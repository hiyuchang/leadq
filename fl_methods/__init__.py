from .fedavg import FedAvg

_method_class_map = {
    'fedavg': FedAvg
}


def get_fl_method_class(key):
    if key in _method_class_map:
        return _method_class_map[key]
    else:
        raise ValueError('Invalid method: {}'.format(key))
