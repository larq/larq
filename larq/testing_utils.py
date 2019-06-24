import larq as lq

from tensorflow import keras


def get_small_bnn_model(input_dim, num_hidden, output_dim):
    model = keras.models.Sequential()
    model.add(
        lq.layers.QuantDense(
            units=num_hidden,
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            activation="relu",
            input_shape=(input_dim,),
            use_bias=False,
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(
        lq.layers.QuantDense(
            units=output_dim,
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            input_quantizer="ste_sign",
            activation="softmax",
            use_bias=False,
        )
    )
    return model
