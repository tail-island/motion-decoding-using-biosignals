import numpy as np
import pickle

from scipy.io import loadmat


def write_parameter(data):
    for user_id, user_data in data:
        xs, ys = user_data

        with open(f'../input/dataset/{user_id:04}-parameter.pickle', mode='wb') as f:
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


def write_dataset(data):
    for user_id, user_data in data:
        with open(f'../input/dataset/{user_id:04}-parameter.pickle', mode='rb') as f:
            parameter = pickle.load(f)

        xs, ys = user_data

        xs = np.transpose(xs, (0, 2, 1))
        ys = np.transpose(ys, (0, 2, 1))

        # xs = (xs - parameter['xs_min']) / (parameter['xs_max'] - parameter['xs_min'])
        # ys = (ys - parameter['ys_min']) / (parameter['ys_max'] - parameter['ys_min'])

        xs = (xs - parameter['xs_mean']) / parameter['xs_std']
        ys = (ys - parameter['ys_mean']) / parameter['ys_std']

        np.save(f'../input/dataset/{user_id:04}-xs.npy', xs)
        np.save(f'../input/dataset/{user_id:04}-ys.npy', ys)


mat = loadmat('../input/motion-decoding-using-biosignals/train.mat')
data = tuple(map(
    lambda i: (
        i,
        (
            mat[f'{i:04}'][0][0][0],
            mat[f'{i:04}'][0][0][1]
        )
    ),
    range(1, 4 + 1)
))


write_parameter(data)
write_dataset(data)

mat = loadmat('../input/motion-decoding-using-biosignals/reference.mat')
data = tuple(map(
    lambda i: (
        i,
        (
            mat[f'{i:04}'][0][0][0],
            mat[f'{i:04}'][0][0][1]
        )
    ),
    (5,)
))

write_parameter(data)
write_dataset(data)
