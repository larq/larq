<img src="https://user-images.githubusercontent.com/13285808/66865479-39c3b600-ef8f-11e9-9bd4-d47b8e432140.gif" alt="logo" height="100px" align="left"/>
<br/>

[![](https://github.com/larq/larq/workflows/Unittest/badge.svg)](https://github.com/larq/larq/actions?workflow=Unittest) [![Codecov](https://img.shields.io/codecov/c/github/larq/larq)](https://codecov.io/github/larq/larq?branch=main) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/larq.svg)](https://pypi.org/project/larq/) [![PyPI](https://img.shields.io/pypi/v/larq.svg)](https://pypi.org/project/larq/) [![PyPI - License](https://img.shields.io/pypi/l/larq.svg)](https://github.com/larq/larq/blob/main/LICENSE) [![DOI](https://joss.theoj.org/papers/10.21105/joss.01746/status.svg)](https://doi.org/10.21105/joss.01746) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Larq is an open-source deep learning library for training neural networks with extremely low precision weights and activations, such as Binarized Neural Networks (BNNs).

Existing deep neural networks use 32 bits, 16 bits or 8 bits to encode each weight and activation, making them large, slow and power-hungry.
This prohibits many applications in resource-constrained environments. Larq is the first step towards solving this. It is designed to provide an easy to use, composable way to train BNNs (1 bit) and other types of Quantized Neural Networks (QNNs) and is based on the `tf.keras` interface. Note that efficient inference using a trained BNN requires the use of an optimized inference engine; we provide these for several platforms in [Larq Compute Engine](https://github.com/larq/compute-engine).

_Larq is part of a family of libraries for BNN development; you can also check out [Larq Zoo](https://github.com/larq/zoo) for pretrained models and [Larq Compute Engine](https://github.com/larq/compute-engine) for deployment on mobile and edge devices._

## Getting Started

To build a QNN, Larq introduces the concept of [quantized layers](https://docs.larq.dev/larq/api/layers/) and [quantizers](https://docs.larq.dev/larq/api/quantizers/). A quantizer defines the way of transforming a full precision input to a quantized output and the pseudo-gradient method used for the backwards pass. Each quantized layer requires an `input_quantizer` and a `kernel_quantizer` that describe the way of quantizing the incoming activations and weights of the layer respectively. If both `input_quantizer` and `kernel_quantizer` are `None` the layer is equivalent to a full precision layer.

You can define a simple binarized fully-connected Keras model using the [Straight-Through Estimator](https://docs.larq.dev/larq/api/quantizers/#ste_sign) the following way:

```python
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(),
        larq.layers.QuantDense(
            512, kernel_quantizer="ste_sign", kernel_constraint="weight_clip"
        ),
        larq.layers.QuantDense(
            10,
            input_quantizer="ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            activation="softmax",
        ),
    ]
)
```

This layer can be used inside a [Keras model](https://www.tensorflow.org/guide/keras/overview#sequential_model) or with a [custom training loop](https://www.tensorflow.org/guide/keras/train_and_evaluate#part_ii_writing_your_own_training_evaluation_loops_from_scratch).

## Examples

Check out our examples on how to train a Binarized Neural Network in just a few lines of code:

- [Introduction to BNNs with Larq](https://docs.larq.dev/larq/tutorials/mnist/)
- [BinaryNet on CIFAR10](https://docs.larq.dev/larq/tutorials/binarynet_cifar10/)

## Installation

Before installing Larq, please install:

- [Python](https://www.python.org/) version `3.7`, `3.8`, `3.9`, or `3.10`
- [Tensorflow](https://www.tensorflow.org/install) version `1.14`, `1.15`, `2.0`, `2.1`, `2.2`, `2.3`, `2.4`, `2.5`, `2.6`, `2.7`, `2.8`, `2.9`, or `2.10`:
  ```shell
  pip install tensorflow  # or tensorflow-gpu
  ```

You can install Larq with Python's [pip](https://pip.pypa.io/en/stable/) package manager:

```shell
pip install larq
```

## About

Larq is being developed by a team of deep learning researchers and engineers at Plumerai to help accelerate both our own research and the general adoption of Binarized Neural Networks.
