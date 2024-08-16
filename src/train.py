import gc
import keras
import numpy as np
import sys

from model import create_model
from multiprocessing import Process
from parameter import BATCH_SIZE, LEARNING_RATE, NUMBER_OF_EPOCHS, NUMBER_OF_MODELS, NUMBER_OF_WARMUP_EPOCHS
from utility import RootMeanSquaredError3D


user_id = int(sys.argv[1])
is_validation = sys.argv[2] == '0'


def pre_train():
    xs = np.concatenate(
        (
            np.load('../input/dataset/0005-xs.npy'),
            *map(
                lambda i: np.load(f'../input/dataset/{i:04}-xs.npy'),
                filter(
                    lambda i: i != user_id,
                    range(1, 4 + 1)
                )
            )
        )
    )
    ys = np.concatenate(
        (
            np.load('../input/dataset/0005-ys.npy'),
            *map(
                lambda i: np.load(f'../input/dataset/{i:04}-ys.npy'),
                filter(
                    lambda i: i != user_id,
                    range(1, 4 + 1)
                )
            )
        )
    )

    validation_data = (
        np.load(f'../input/dataset/{user_id:04}-xs.npy'),
        np.load(f'../input/dataset/{user_id:04}-ys.npy')
    )

    model = create_model()

    model.compile(
        optimizer=keras.optimizers.Lion(
            learning_rate=keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=1e-9,
                decay_steps=(NUMBER_OF_EPOCHS // 2 - NUMBER_OF_WARMUP_EPOCHS) * int(np.ceil(len(xs) / BATCH_SIZE)),
                warmup_target=LEARNING_RATE,
                warmup_steps=NUMBER_OF_EPOCHS // 2 * int(np.ceil(len(xs) / BATCH_SIZE))
            )
        ),
        loss=RootMeanSquaredError3D(5)
    )

    model.fit(
        xs,
        ys,
        batch_size=BATCH_SIZE,
        epochs=NUMBER_OF_EPOCHS // 2,
        validation_data=validation_data,
    )

    model.save('../input/dataset/0000.keras')

    del model
    gc.collect()


def train(model_number):
    xs = np.load(f'../input/dataset/{user_id:04}-xs.npy')
    ys = np.load(f'../input/dataset/{user_id:04}-ys.npy')

    if is_validation:
        validation_data = (xs[:30], ys[:30])

        xs = xs[30:]
        ys = ys[30:]
    else:
        validation_data = None

    model = keras.models.load_model('../input/dataset/0000.keras', {'root_mean_squared_error_3d': RootMeanSquaredError3D(user_id)})

    # if is_validation:
    #     model.summary()

    model.compile(
        optimizer=keras.optimizers.Lion(
            learning_rate=keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=1e-9,
                decay_steps=(NUMBER_OF_EPOCHS - NUMBER_OF_WARMUP_EPOCHS) * int(np.ceil(len(xs) / BATCH_SIZE)),
                warmup_target=LEARNING_RATE,
                warmup_steps=NUMBER_OF_EPOCHS * int(np.ceil(len(xs) / BATCH_SIZE))
            )
        ),
        loss=RootMeanSquaredError3D(user_id)
    )

    model.fit(
        xs,
        ys,
        batch_size=BATCH_SIZE,
        epochs=NUMBER_OF_EPOCHS,
        validation_data=validation_data
    )

    model.save(f'../input/dataset/{user_id:04}-{model_number:02}.keras')

    del model
    gc.collect()


for i in range(1, NUMBER_OF_MODELS + 1):
    p = Process(target=pre_train)
    p.start()
    p.join()

    p = Process(target=lambda: train(i))
    p.start()
    p.join()
