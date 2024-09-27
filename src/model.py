import keras

from funcy import identity, juxt, ljuxt, rcompose
from parameter import DROPOUT_RATE, NOISE_STDDEV


def create_model():
    def Activation():
        return keras.layers.Activation('mish')

    def Add():
        return keras.layers.Add()

    def Conv(filters):
        return keras.layers.Conv1D(filters, 5, padding='same', use_bias=True, kernel_initializer=keras.initializers.HeNormal())

    def DepthwiseConv(depth_multiplier=1):
        return keras.layers.DepthwiseConv1D(5, padding='same', depth_multiplier=depth_multiplier, use_bias=True, depthwise_initializer=keras.initializers.HeNormal())

    def Dropout():
        return keras.layers.Dropout(DROPOUT_RATE)

    def GaussianNoise():
        return keras.layers.GaussianNoise(NOISE_STDDEV)

    def Normalization():
        return keras.layers.LayerNormalization()

    def Pooling():
        return keras.layers.AveragePooling1D(2)

    ####

    def DepthwiseConvUnit0():
        return rcompose(
            Normalization(),
            Activation(),
            ljuxt(
                rcompose(
                    DepthwiseConv(2),

                    Normalization(),
                    Activation(),
                    Dropout(),
                    DepthwiseConv()
                ),
                DepthwiseConv(2)
            ),
            Add()
        )

    def DepthwiseConvUnit():
        return rcompose(
            ljuxt(
                rcompose(
                    Normalization(),
                    Activation(),
                    DepthwiseConv(),

                    Normalization(),
                    Activation(),
                    Dropout(),
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
                    Dropout(),
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
                    Dropout(),
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
            x = DepthwiseConvUnit0()(x)

            for _ in range(3 - 1):
                x = DepthwiseConvUnit()(x)

            x = Pooling()(x)

        x = x[:, :30, :]

        for filters in (256, 128, 64, 32, 3):
            x = ConvUnit0(filters)(x)

            for _ in range(4 - 1):
                x = ConvUnit(filters)(x)

        return x

    return keras.Model(
        *juxt(
            identity,
            op
        )(keras.Input((1000, 16)))
    )
