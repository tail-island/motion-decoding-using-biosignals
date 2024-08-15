import keras

from funcy import identity, juxt, ljuxt, rcompose


def create_model():
    def Activation():
        return keras.layers.Activation('mish')

    def Add():
        return keras.layers.Add()

    def BatchNormalization():
        return keras.layers.BatchNormalization()

    def DepthwiseConv(depth_multiplier=1):
        return keras.layers.DepthwiseConv1D(3, padding='same', depth_multiplier=depth_multiplier, use_bias=False, depthwise_initializer=keras.initializers.HeNormal())

    def Dense(units):
        return keras.layers.Dense(units, use_bias=False, kernel_initializer=keras.initializers.HeNormal())

    def Dropout(rate):
        return keras.layers.Dropout(rate)

    def Flatten():
        return keras.layers.Flatten()

    def GaussianNoise():
        return keras.layers.GaussianNoise(1e-3)

    def LayerNormalization():
        return keras.layers.LayerNormalization()

    def Pooling():
        return keras.layers.AveragePooling1D(2)

    def Reshape(shape):
        return keras.layers.Reshape(shape)

    ####

    def ConvUnit():
        return rcompose(
            ljuxt(
                rcompose(
                    LayerNormalization(),
                    Activation(),
                    DepthwiseConv(),

                    LayerNormalization(),
                    Activation(),
                    Dropout(0.3),
                    DepthwiseConv()
                ),
                identity
            ),
            Add()
        )

    def MlpUnit(units):
        return rcompose(
            ljuxt(
                rcompose(
                    BatchNormalization(),
                    Activation(),
                    Dense(units),

                    BatchNormalization(),
                    Activation(),
                    Dropout(0.3),
                    Dense(units)
                ),
                identity
            ),
            Add()
        )

    ###

    def op(x):
        x = GaussianNoise()(x)

        for _ in range(4):
            x = DepthwiseConv(2)(x)

            for _ in range(2):
                x = ConvUnit()(x)

            x = Pooling()(x)

        x = Flatten()(x)

        for units in (8192, 4096, 1024, 512):
            x = Dense(units)(x)

            for _ in range(2):
                x = MlpUnit(units)(x)

        x = Dense(30 * 3)(x)
        x = Reshape((30, 3))(x)

        return x

    return keras.Model(*juxt(identity, op)(keras.Input((1000, 16))))
