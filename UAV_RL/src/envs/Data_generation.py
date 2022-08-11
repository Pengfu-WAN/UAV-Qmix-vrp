from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import pandas as pd


def get_data(file_name):
    dir_name = os.path.dirname(os.path.realpath('__file__'))
    print(dir_name)
    file_name = os.path.join(dir_name, file_name)
    df = pd.read_csv(file_name, encoding='latin1')
    customer_size = df.shape[0] - 1
    Node = [df['Node'][i] for i in range(customer_size + 1)]
    Local_X = [df['Local_X'][i] for i in range(customer_size + 1)]
    Local_Y = [df['Local_Y'][i] for i in range(customer_size + 1)]
    alpha = [df['alpha'][i] for i in range(customer_size + 1)]
    beta = [df['beta'][i] for i in range(customer_size + 1)]
    base = [df['base'][i] for i in range(customer_size + 1)]
    Capacity = df['Capacity'][0]
    Velocity = df['Velocity'][0]
    UAV_num = df['UAV_num'][0]
    return Node, Local_X, Local_Y, alpha, beta, base, Capacity, Velocity, UAV_num

class VRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=10, offset=0, seed_value=np.arange(10), distribution=None):
        super(VRPDataset, self).__init__()

        self.data = []
        if filename is not None:
            Node, Local_X, Local_Y, alpha, beta, base, Capacity, Velocity, UAV_num = get_data(filename)
            loc = np.concatenate((np.expand_dims(Local_X, axis=1), np.expand_dims(Local_Y, axis=1)), axis=1)
            self.data.append(
                {
                    'UAV_num': int(UAV_num),
                    'loc': np.array(loc[1:]),
                    'alpha': np.array(alpha[1:]),
                    'beta': np.array(beta[1:]),
                    'base': np.array(base[1:]),
                    'velocity': Velocity,
                    'capacity': Capacity,
                    'depot': np.array(loc[0])
                }
            )

        else:
            self.data = []
            for i in range(num_samples):
                np.random.seed(seed_value[i])
                self.data.append(
                    {
                        'UAV_num': np.random.randint(low=5, high=10),
                        'loc': np.random.random_sample((size, 2)),
                        'alpha': np.random.random_sample(size) * 0.004 + 0.001,
                        'beta': np.random.random_sample(size) * 20 + 10,
                        'base': np.random.random_sample(size) * 4 + 6,
                        'velocity': 0.01 * np.random.rand() + 0.05,
                        'capacity': np.random.randint(low=110, high=120),
                        'depot': np.random.random_sample(2)
                    }
                )

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
