<h1><img src="/images/logo.svg" alt="logo" width="450px" style="display: block; margin-left: auto; margin-right: auto"/></h1>

Larq is an open-source deep learning library for training neural networks with extremely low-precision weights and activations, such as Binarized Neural Networks (BNNs)[^1].

Existing deep neural networks use 32 bits, 16 bits or 8 bits to encode each weight and activation, making them large, slow and power-hungry. This prohibits many applications in resource-constrained environments. Larq is the first step towards solving this. It is designed to provide an easy to use, composable way to train BNNs (1 bit) and other types of Quantized Neural Networks (QNNs) and is based on the `tf.keras` interface.

## Getting Started

To build a QNN, Larq introduces the concept of [quantized layers](https://larq.dev/api/layers/) and [quantizers](https://larq.dev/api/quantizers/). A quantizer defines the way of transforming a full precision input to a quantized output and the pseudo-gradient method used for the backwards pass. Each quantized layer requires an `input_quantizer` and a `kernel_quantizer` that describe the way of quantizing the incoming activations and weights of the layer respectively. If both `input_quantizer` and `kernel_quantizer` are `None` the layer is equivalent to a full precision layer.

You can define a simple binarized fully-connected Keras model using the [Straight-Through Estimator](https://larq.dev/api/quantizers/#ste_sign) the following way:

```python
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

This layer can be used inside a [Keras model](https://www.tensorflow.org/alpha/guide/keras/overview#sequential_model) or with a [custom training loop](https://www.tensorflow.org/alpha/guide/keras/overview#model_subclassing).

## Installation

Before installing Larq, please install:

- [Python](https://python.org) version `3.6` or `3.7`
- [Tensorflow](https://www.tensorflow.org/install) version `1.13`, `1.14`, `1.15` or `2.0.0`:
  ```shell
  pip install tensorflow  # or tensorflow-gpu
  ```

You can install Larq with Python's [pip](https://pip.pypa.io/en/stable/) package manager:

```shell
pip install larq
```

[^1]: Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., & Bengio, Y. (2016). <a href="http://papers.nips.cc/paper/6573-binarized-neural-networks.pdf" target="_blank" >Binarized Neural Networks</a>. In Advances in neural information processing systems (pp. 4107-4115).
