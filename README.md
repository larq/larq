# Larq

[![Azure DevOps builds](https://img.shields.io/azure-devops/build/plumerai/larq/14.svg?logo=azure-devops)](https://plumerai.visualstudio.com/larq/_build/latest?definitionId=14&branchName=master) [![Azure DevOps coverage](https://img.shields.io/azure-devops/coverage/plumerai/larq/14.svg?logo=azure-devops)](https://plumerai.visualstudio.com/larq/_build/latest?definitionId=14&branchName=master) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/larq.svg)](https://pypi.org/project/larq/) [![PyPI](https://img.shields.io/pypi/v/larq.svg)](https://pypi.org/project/larq/) [![PyPI - License](https://img.shields.io/pypi/l/larq.svg)](https://github.com/larq/larq/blob/master/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/larq/larq/master?filepath=docs%2Fexamples) [![Join the community on Spectrum](https://withspectrum.github.io/badge/badge.svg)](https://spectrum.chat/larq)

Larq is an open-source deep learning library for training neural networks with extremely low precision weights and activations, such as Binarized Neural Networks (BNNs).

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

## Examples

Check out our examples on how to train a Binarized Neural Network in just a few lines of code:

- [Introduction to BNNs with Larq](https://larq.dev/examples/mnist/)
- [BinaryNet on CIFAR10](https://larq.dev/examples/binarynet_cifar10/)
- [BinaryNet on CIFAR10 (Advanced)](https://larq.dev/examples/binarynet_advanced_cifar10/)

## Requirements

Before installing Larq, please install:

- [Python](https://python.org) version `3.6` or `3.7`
- [Tensorflow](https://www.tensorflow.org/install) version `1.13+` or `2.0.0`

You can also check out one of our prebuilt [docker images](https://hub.docker.com/r/plumerai/deep-learning/tags).

## Installation

You can install Larq with Python's [pip](https://pip.pypa.io/en/stable/) package manager:

```shell
pip install larq
```
