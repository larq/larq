import numpy as np
import tensorflow as tf
import larq as lq


def assert_weights(weights, expected):
    for w, e in zip(weights, expected):
        np.testing.assert_allclose(np.squeeze(w), e)


def test_xavier_scaling():
    # TODO: fix compatibility with TF 2.0
    if int(tf.__version__[0]) == 2:
        return
    dense = lq.layers.QuantDense(
        1, kernel_quantizer="ste_sign", kernel_initializer="zeros", input_shape=(1,)
    )
    model = tf.keras.models.Sequential([dense])

    model.compile(
        loss="mae",
        optimizer=lq.optimizers.XavierLearningRateScaling(
            model, tf.keras.optimizers.SGD, lr=1
        ),
    )
    assert_weights(dense.get_weights(), [0, 0])
    model.fit(np.array([1.0]), np.array([2.0]), epochs=1, batch_size=1)
    assert_weights(dense.get_weights(), [1 / np.sqrt(1.5 / 2), 1])
