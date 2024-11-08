import keras
# import matplotlib.pyplot as plot
import numpy as np
import pickle
import sys

from scipy.stats import trim_mean
from utility import RootMeanSquaredError3D


user_id = int(sys.argv[1])
number_of_models = 10


with open(f'../input/dataset/{user_id:04}-parameter.pickle', mode='rb') as f:
    parameter = pickle.load(f)

xs = np.load(f'../input/dataset/{user_id:04}-xs.npy')[:30]

pred_ys = trim_mean(
    np.array(tuple(map(
        lambda i: keras.models.load_model(f'../input/dataset/{user_id:04}-{i:02}.keras', {'root_mean_squared_error_3d': RootMeanSquaredError3D(user_id)}).predict(xs, batch_size=32, verbose=False),
        range(1, number_of_models + 1)
    ))),
    0.4,
    axis=0
)

true_ys, pred_ys = map(
    lambda ys: ys * parameter['ys_std'] + parameter['ys_mean'],
    (
        np.load(f'../input/dataset/{user_id:04}-ys.npy')[:30],
        pred_ys
    )
)


# for true_y, pred_y in zip(true_ys, pred_ys):
#     figure, axes = plot.subplots(1, 3, figsize=(30, 10))

#     for j in range(3):
#         axes[j].plot(np.poly1d(np.polyfit(np.arange(30), pred_y[:, j], 3))(np.arange(30)))
#         axes[j].plot(pred_y[:, j])
#         axes[j].plot(true_y[:, j])
#         axes[j].set_ylim(-5, 5)

#     plot.show()


for i in range(len(pred_ys)):
    for j in range(3):
        pred_ys[i, :, j] = np.poly1d(np.polyfit(np.arange(30), pred_ys[i, :, j], 3))(np.arange(30))


print(np.sqrt(np.mean(np.sum((true_ys - pred_ys) ** 2, axis=2))))
