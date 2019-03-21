import tensorflow as tf
import tensorflow_datasets as tfds

import xquant as xq


model = tf.keras.models.Sequential(
    [
        xq.layers.QuantConv2D(
            128,
            3,
            kernel_quantizer="sign_clip_ste",
            use_bias=False,
            input_shape=(32, 32, 3),
        ),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
        xq.layers.QuantConv2D(
            128,
            3,
            kernel_quantizer="sign_clip_ste",
            input_quantizer="sign_clip_ste",
            padding="same",
            use_bias=False,
        ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
        xq.layers.QuantConv2D(
            256,
            3,
            kernel_quantizer="sign_clip_ste",
            input_quantizer="sign_clip_ste",
            padding="same",
            use_bias=False,
        ),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
        xq.layers.QuantConv2D(
            256,
            3,
            kernel_quantizer="sign_clip_ste",
            input_quantizer="sign_clip_ste",
            padding="same",
            use_bias=False,
        ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
        xq.layers.QuantConv2D(
            512,
            3,
            kernel_quantizer="sign_clip_ste",
            input_quantizer="sign_clip_ste",
            padding="same",
            use_bias=False,
        ),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
        xq.layers.QuantConv2D(
            512,
            3,
            kernel_quantizer="sign_clip_ste",
            input_quantizer="sign_clip_ste",
            padding="same",
            use_bias=False,
        ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
        tf.keras.layers.Flatten(),
        xq.layers.QuantDense(
            1024,
            kernel_quantizer="sign_clip_ste",
            input_quantizer="sign_clip_ste",
            use_bias=False,
        ),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
        xq.layers.QuantDense(
            1024,
            kernel_quantizer="sign_clip_ste",
            input_quantizer="sign_clip_ste",
            use_bias=False,
        ),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
        xq.layers.QuantDense(
            10,
            kernel_quantizer="sign_clip_ste",
            input_quantizer="sign_clip_ste",
            use_bias=False,
        ),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
        tf.keras.layers.Activation("softmax"),
    ]
)

batch_size = 32
learning_rate = 1e-3


def preprocess(data):
    return data["image"] / 255, tf.one_hot(data["label"], 10)


train_data, eval_data = tfds.load(name="cifar10", split=["train", "test"])
train_data = (
    train_data.shuffle(1000).repeat().map(preprocess).batch(batch_size).prefetch(1)
)
eval_data = eval_data.repeat().map(preprocess).batch(batch_size).prefetch(1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


model.fit(
    train_data,
    epochs=100,
    steps_per_epoch=50000 // batch_size,
    validation_data=eval_data,
    validation_steps=10000 // batch_size,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir="logs")],
)
