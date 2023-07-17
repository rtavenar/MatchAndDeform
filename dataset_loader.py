import numpy as np
import torch


def from_numpy_to_torch(filename, float_or_long=True):
    data = np.load(filename)
    data_t = torch.from_numpy(data)
    if float_or_long:
        data_t = data_t.type(torch.float)
    else:
        data_t = data_t.type(torch.long)
    return data_t


def load_a_dataset(dataset_name, domain_id):

    if type(domain_id) is not str:
        domain_id = str(domain_id)

    train = from_numpy_to_torch(dataset_name + "_" + domain_id + 'train.npy')
    train_label = from_numpy_to_torch(dataset_name +  "_" + domain_id + 'train_labels.npy', float_or_long=False)

    valid = from_numpy_to_torch(dataset_name +  "_" + domain_id + 'valid.npy')
    valid_label = from_numpy_to_torch(dataset_name +  "_" + domain_id + 'valid_labels.npy', float_or_long=False)

    test = from_numpy_to_torch(dataset_name +  "_" + domain_id + 'test.npy')
    test_label = from_numpy_to_torch(dataset_name +  "_" + domain_id + 'test_labels.npy', float_or_long=False)

    return train, train_label, valid, valid_label, test, test_label