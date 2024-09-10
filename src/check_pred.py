import matplotlib.pyplot as plot
import numpy as np
import pickle

from sklearn.decomposition import PCA


for user_id in range(1, 5 + 1):
    with open(f'../input/dataset/{user_id:04}-parameter.pickle', mode='rb') as f:
        parameter = pickle.load(f)

    figure, axes = plot.subplots(2, 2, figsize=(20, 20))

    v_1 = np.reshape(np.load(f'../input/dataset/{user_id:04}-ys.npy') * parameter['ys_std'] + parameter['ys_mean'], [-1, 3])[:, :2]
    # v = v[:1800]

    pca_1 = PCA(2)
    pca_1.fit(v_1)

    axes[0, 0].scatter(v_1[:, 0], v_1[:, 1], alpha=0.05)
    axes[0, 0].plot([0, pca_1.components_[0, 0]], [0, pca_1.components_[0, 1]], color='red')
    axes[0, 0].plot(0, 0, 'ro')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_xlim([-6, 6])
    axes[0, 0].set_ylim([-6, 6])

    theta = np.arctan2(pca_1.components_[0, 1], pca_1.components_[0, 0])

    v_1 = np.stack(
        [
            v_1[:, 0] * np.cos(-theta) - v_1[:, 1] * np.sin(-theta),
            v_1[:, 0] * np.sin(-theta) + v_1[:, 1] * np.cos(-theta)
        ],
        axis=1
    )

    axes[1, 0].scatter(v_1[:, 0], v_1[:, 1], alpha=0.05)
    axes[1, 0].plot(0, 0, 'ro')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_xlim([-6, 6])
    axes[1, 0].set_ylim([-6, 6])

    v_2 = np.load(f'../input/dataset/{user_id:04}-pred-ys.npy')

    if user_id == 5:
        v_2 = v_2 * parameter['ys_std'] + parameter['ys_mean']

    v_2 = np.reshape(v_2, [-1, 3])[:, :2]

    pca_2 = PCA(2)
    pca_2.fit(v_2)

    axes[0, 1].scatter(v_2[:, 0] * -1, v_2[:, 1] * -1, alpha=0.05)
    axes[0, 1].plot([0, pca_2.components_[0, 0]], [0, pca_2.components_[0, 1]], color='red')
    axes[0, 1].plot(0, 0, 'ro')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_xlim([-6, 6])
    axes[0, 1].set_ylim([-6, 6])

    theta = np.arctan2(pca_2.components_[0, 1], pca_2.components_[0, 0])

    v_2 = np.stack(
        [
            v_2[:, 0] * np.cos(-theta) - v_2[:, 1] * np.sin(-theta),
            v_2[:, 0] * np.sin(-theta) + v_2[:, 1] * np.cos(-theta)
        ],
        axis=1
    )

    axes[1, 1].scatter(v_2[:, 0] * -1, v_2[:, 1] * -1, alpha=0.05)
    axes[1, 1].plot(0, 0, 'ro')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_xlim([-6, 6])
    axes[1, 1].set_ylim([-6, 6])

    plot.show()
