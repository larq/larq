Larq is an open-source Python library for training neural networks with extremely low-precision weights and activations, such as Binarized Neural Networks (BNNs)[^1].

Existing deep neural networks use 32 bits, 16 bits or 8 bits to encode each weight and activation, making them large, slow and power-hungry. This prohibits many applications in resource-constrained environments. Larq is the first step towards solving this.
The API of Larq is built on top of `tf.keras` and is designed to provide an easy to use, composable way to design and train BNNs (1 bit) and other types of Quantized Neural Networks (QNNs). It provides tools specifically designed to aid in BNN development, such as specialized optimizers, training metrics, and profiling tools.
It is aimed at both researchers in the field of efficient deep learning and practitioners who want to explore BNNs for their applications. Furthermore, Larq makes it easier for beginners and students to get started with the field of efficient deep learning.

To build a QNN, Larq introduces the concept of [quantized layers](/larq/api/layers/) and [quantizers](/larq/api/quantizers/). A quantizer defines the way of transforming a full precision input to a quantized output and the pseudo-gradient method used for the backwards pass. Each quantized layer requires an `input_quantizer` and a `kernel_quantizer` that describe the way of quantizing the incoming activations and weights of the layer respectively. If both `input_quantizer` and `kernel_quantizer` are `None` the layer is equivalent to a full precision layer. This layer can be used inside a [Keras model](https://www.tensorflow.org/guide/keras/overview#sequential_model) or with a [custom training loop](https://www.tensorflow.org/guide/keras/train_and_evaluate#part_ii_writing_your_own_training_evaluation_loops_from_scratch).

For a detailed explanation checkout our [user guide](/larq/guides/key-concepts/).

## Defining a simple BNN

A simple fully-connected BNN using the [Straight-Through Estimator](/larq/api/quantizers/#ste_sign) can be defined in just a few lines of code using either the Keras sequential, functional or model subclassing APIs:

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

## Installation

Before installing Larq, please install:

- [Python](https://www.python.org) version `3.6` or `3.7`
- [Tensorflow](https://www.tensorflow.org/install) version `1.14`, `1.15` or `2.0.0`:
  ```shell
  pip install tensorflow  # or tensorflow-gpu
  ```

You can install Larq with Python's [pip](https://pip.pypa.io/en/stable/) package manager:

```shell
pip install larq
```

[^1]: Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., & Bengio, Y. (2016). <a href="http://papers.nips.cc/paper/6573-binarized-neural-networks.pdf" target="_blank" >Binarized Neural Networks</a>. In Advances in neural information processing systems (pp. 4107-4115).
