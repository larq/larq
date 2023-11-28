from larq import (  # pytype: disable=pyi-error
    activations,
    callbacks,
    constraints,
    context,
    layers,
    math,
    metrics,
    models,
    optimizers,
    quantizers,
    utils,
)
from larq import layers
import keras

model = keras.models.Sequential(
    [
        keras.layers.Flatten(),
        layers.QuantDense(
            512, kernel_quantizer="ste_sign", kernel_constraint="weight_clip"
        ),
        layers.QuantDense(
            10,
            input_quantizer="ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            activation="softmax",
        ),
    ]
)

dummy_input = keras.Input(shape=(1, 512))  # Replace with your actual input shape
model(dummy_input)
model.summary()

keras.models.save_model(model, 'larq_model.h5')

