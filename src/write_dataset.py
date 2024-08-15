import numpy as np
import pickle

from scipy.io import loadmat


def write_parameter(train):
    for i in range(1, 4 + 1):
        xs, ys, _, _ = train[f'{i:04}'][0][0]

        with open(f'../input/dataset/{i:04}-parameter.pickle', mode='wb') as f:
            pickle.dump(
                {
                    'xs_max': np.max(xs),
                    'xs_min': np.min(xs),
                    'xs_mean': np.mean(xs),
                    'xs_std': np.std(xs),
                    'ys_max': np.max(ys),
                    'ys_min': np.min(ys),
                    'ys_mean': np.mean(ys),
                    'ys_std': np.std(ys)
                },
                f
            )


def write_dataset(train):
    for i in range(1, 4 + 1):
        with open(f'../input/dataset/{i:04}-parameter.pickle', mode='rb') as f:
            parameter = pickle.load(f)

        xs, ys, _, _ = train[f'{i:04}'][0][0]

        xs = np.transpose(xs, (0, 2, 1))
        ys = np.transpose(ys, (0, 2, 1))

        # xs = (xs - parameter['xs_min']) / (parameter['xs_max'] - parameter['xs_min'])
        # ys = (ys - parameter['ys_min']) / (parameter['ys_max'] - parameter['ys_min'])

        xs = (xs - parameter['xs_mean']) / parameter['xs_std']
        ys = (ys - parameter['ys_mean']) / parameter['ys_std']

        np.save(f'../input/dataset/{i:04}-xs.npy', xs)
        np.save(f'../input/dataset/{i:04}-ys.npy', ys)


train = loadmat('../input/motion-decoding-using-biosignals/train.mat')

write_parameter(train)
write_dataset(train)
