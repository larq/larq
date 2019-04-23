# User Guide

To build a Quantized Neural Network (QNN), Larq introduces the concept of [quantized layers](https://plumerai.github.io/larq/api/layers/) and [quantizers](https://plumerai.github.io/larq/api/quantizers/). A quantizer defines the way of transforming a full precision input to a quantized output and the pseudo-gradient method used for the backwards pass.

Each quantized layer requires a `kernel_quantizer` and an `input_quantizer` that describe the way of quantizing the weights of the layer and the activations of the previous layer respectively. If both `input_quantizer` and `kernel_quantizer` are `None` the layer is equivalent to a full precision layer. Larq layers are fully compatible with the Keras API so you can use them with Keras Layers interchangeably:

```python tab="Larq 32-bit model"
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    larq.layers.QuantDense(512, activation="relu"),
    larq.layers.QuantDense(10, activation="softmax")
])
```

```python tab="Keras 32-bit model"
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

A simple fully-connected Binarized Neural Network (BNN) using the [Straight-Through Estimator](https://plumerai.github.io/larq/api/quantizers/#ste_sign) can be defined in just a few lines of code using either the Keras sequential, functional or model subclassing APIs:

```python tab="Larq 1-bit model"
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    larq.layers.QuantDense(512,
                           kernel_quantizer="ste_sign",
                           kernel_constraint="weight_clip"),
    larq.layers.QuantDense(10,
                           input_quantizer="ste_sign",
                           kernel_quantizer="ste_sign",
                           kernel_constraint="weight_clip",
                           activation="softmax")])
```

```python tab="Larq 1-bit model functional"
x = tf.keras.Input(shape=(28, 28, 1))
y = tf.keras.layers.Flatten()(x)
y = larq.layers.QuantDense(512,
                           kernel_quantizer="ste_sign",
                           kernel_constraint="weight_clip")(y)
y = larq.layers.QuantDense(10,
                           input_quantizer="ste_sign",
                           kernel_quantizer="ste_sign",
                           kernel_constraint="weight_clip",
                           activation="softmax")(y)
model = tf.keras.Model(inputs=x, outputs=y)
```

```python tab="Larq 1-bit model subclassing"
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = larq.layers.QuantDense(512,
                                             kernel_quantizer="ste_sign",
                                             kernel_constraint="weight_clip")
        self.dense2 = larq.layers.QuantDense(10,
                                             input_quantizer="ste_sign",
                                             kernel_quantizer="ste_sign",
                                             kernel_constraint="weight_clip",
                                             activation="softmax")

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)

model = MyModel()
```
