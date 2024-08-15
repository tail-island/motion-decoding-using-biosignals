import keras
import pickle


def RootMeanSquaredError3D(user_id):
    def root_mean_squared_error_3d(y_true, y_pred):
        y_true, y_pred = map(
            lambda ys: ys * parameter['ys_std'] + parameter['ys_mean'],
            (
                y_true,
                y_pred
            )
        )

        return keras.ops.sqrt(keras.ops.mean(keras.ops.sum((y_true - y_pred) ** 2, axis=2)))

    with open(f'../input/dataset/{user_id:04}-parameter.pickle', mode='rb') as f:
        parameter = pickle.load(f)

    return root_mean_squared_error_3d
