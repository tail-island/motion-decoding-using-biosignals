import json
import keras
import numpy as np
import pickle

from funcy import count
from itertools import starmap
from parameter import BATCH_SIZE
from scipy.io import loadmat
from scipy.stats import trim_mean
from sklearn.decomposition import PCA
from utility import RootMeanSquaredError3D


number_of_models = 40


test = loadmat('../input/motion-decoding-using-biosignals/test.mat')


def get_sub(user_id):
    with open(f'../input/dataset/{user_id:04}-parameter.pickle', mode='rb') as f:
        parameter = pickle.load(f)

    xs, _, _ = test[f'{user_id:04}'][0][0]

    xs = np.transpose(xs, (0, 2, 1))
    xs = (xs - parameter['xs_min']) / (parameter['xs_max'] - parameter['xs_min'])
    # xs = (xs - parameter['xs_mean']) / parameter['xs_std']

    ys = trim_mean(
        np.array(tuple(map(
            lambda i: keras.models.load_model(f'../input/dataset/{user_id:04}-{i:02}.keras', {'root_mean_squared_error_3d': RootMeanSquaredError3D(user_id)}).predict(xs, batch_size=BATCH_SIZE, verbose=False),
            range(1, number_of_models + 1)
        ))),
        0.4,
        axis=0
    )

    ys = ys * parameter['ys_std'] + parameter['ys_mean']

    for i in range(len(ys)):
        for j in range(3):
            ys[i, :, j] = np.poly1d(np.polyfit(np.arange(30), ys[i, :, j], 3))(np.arange(30))

    np.save(f'../input/dataset/{user_id:04}-pred-ys.npy', ys)

    # # ここから。予測結果を可視化したらなんか横に間延びしていたので、小手先で調整します

    # v = np.reshape(ys, [-1, 3])[:, :2]

    # pca = PCA(2)
    # pca.fit(v)

    # theta = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])

    # v = np.stack(
    #     [
    #         v[:, 0] * np.cos(-theta) - v[:, 1] * np.sin(-theta),
    #         v[:, 0] * np.sin(-theta) + v[:, 1] * np.cos(-theta)
    #     ],
    #     axis=1
    # )

    # match user_id:
    #     case 1:
    #         v[:, 0] = v[:, 0] * 0.8
    #     case 2:
    #         v[:, 0] = v[:, 0] * 0.8
    #     case 3:
    #         v[:, 0] = v[:, 0] * 0.75
    #     case 4:
    #         v[:, 0] = v[:, 0] * 0.7

    # v = np.stack(
    #     [
    #         v[:, 0] * np.cos(theta) - v[:, 1] * np.sin(theta),
    #         v[:, 0] * np.sin(theta) + v[:, 1] * np.cos(theta)
    #     ],
    #     axis=1
    # )

    # ys[:, :, :2] = np.reshape(v, [np.shape(ys)[0], 30, 2])

    # # ここまで

    return (
        f'sub{user_id}',
        dict(starmap(
            lambda i, y: (
                f'trial{i}',
                (y * (-1, -1, 1)).tolist()
            ),
            zip(
                count(1),
                ys
            )
        ))
    )


submission = dict(map(get_sub, range(1, 4 + 1)))


with open('submission.json', mode='w') as f:
    json.dump(submission, f)
