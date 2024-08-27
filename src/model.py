import keras

from funcy import identity, juxt, ljuxt, rcompose


def create_model():
    def Activation():
        return keras.layers.Activation('mish')

    def Add():
        return keras.layers.Add()

    def Conv(filters):
        return keras.layers.Conv1D(filters, 5, padding='same', use_bias=False, kernel_initializer=keras.initializers.HeNormal())

    def DepthwiseConv(depth_multiplier=1):
        return keras.layers.DepthwiseConv1D(3, padding='same', depth_multiplier=depth_multiplier, use_bias=False, depthwise_initializer=keras.initializers.HeNormal())

    def Dropout(rate):
        return keras.layers.Dropout(rate)

    def GaussianNoise():
        return keras.layers.GaussianNoise(1e-3)

    def Normalization():
        return keras.layers.LayerNormalization()

    def Pooling():
        return keras.layers.AveragePooling1D(2)

    ####

    def DepthwiseConvUnit():
        return rcompose(
            ljuxt(
                rcompose(
                    Normalization(),
                    Activation(),
                    DepthwiseConv(),

                    Normalization(),
                    Activation(),
                    Dropout(0.3),
                    DepthwiseConv()
                ),
                identity
            ),
            Add()
        )

    def ConvUnit0(filters):
        return rcompose(
            Normalization(),
            Activation(),
            ljuxt(
                rcompose(
                    Conv(filters),

                    Normalization(),
                    Activation(),
                    Dropout(0.2),
                    Conv(filters)
                ),
                Conv(filters)
            ),
            Add()
        )

    def ConvUnit(filters):
        return rcompose(
            ljuxt(
                rcompose(
                    Normalization(),
                    Activation(),
                    Conv(filters),

                    Normalization(),
                    Activation(),
                    Dropout(0.2),
                    Conv(filters)
                ),
                identity
            ),
            Add()
        )

    ###

    def op(x):
        x = GaussianNoise()(x)

        for _ in range(5):
            x = Normalization()(x)
            x = Activation()(x)
            x = DepthwiseConv(2)(x)

            for _ in range(2):
                x = DepthwiseConvUnit()(x)

            x = Pooling()(x)

        x = x[:, :30, :]

        for filters in (256, 128, 64, 32, 3):
            x = ConvUnit0(filters)(x)

            for _ in range(4 - 1):
                x = ConvUnit(filters)(x)

        return x

    return keras.Model(*juxt(identity, op)(keras.Input((1000, 16))))
