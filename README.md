# xquant

[![Azure DevOps builds](https://img.shields.io/azure-devops/build/plumerai/xquant/4.svg?logo=azure-devops)](https://plumerai.visualstudio.com/xquant/_build/latest?definitionId=4&branchName=master) [![Azure DevOps coverage](https://img.shields.io/azure-devops/coverage/plumerai/xquant/4.svg?logo=azure-devops)](https://plumerai.visualstudio.com/xquant/_build/latest?definitionId=4&branchName=master) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

XQuant is an Open Source Machine Learning Library for Training Super Quantized Neural Networks (SQNNs). Existing Deep Neural Networks tend to be large, slow and power-hungry, prohibiting many applications in resource-constrained environments. XQuant is designed to provide an easy to use, composable way to train SQNNs (e.g. Binarized Neural Networks) based on the `tf.keras` interface.

## Getting Started

To build a SQNN XQuant introduces the concept of _Quantized Layers_ and _Quantizers_. A _Quantizer_ defines the way of transforming a full precision input to a quantized output and the pseudo-gradient method used for the backwards pass. Each _Quantized Layer_ requires a `input_quantizer` and `kernel_quantizer` that describes the way of quantizing the activation of the previous layer and the weights respectively. If both `input_quantizer` and `kernel_quantizer` are `None` the layer is equivalent to a full precision layer.

You can define a binarized densly-connected using the Straight-Through Estimator the following way:

```python
xquant.layers.QuantDense(
    32,
    input_quantizer="ste_sign",
    kernel_quantizer="ste_sign",
    kernel_constraint="weight_clip",
)
```

This layer can be used inside a [keras model](https://www.tensorflow.org/alpha/guide/keras/overview#sequential_model) or with a [custom training loop](https://www.tensorflow.org/alpha/guide/keras/overview#model_subclassing).

Checkout our Example on how to train a Binarized Neural Network in just a few lines of code.

## Requirements

Before installing XQuant, please install:

- [Python](https://python.org) version `3.6` or `3.7`
- [Tensorflow](https://www.tensorflow.org/install) version `1.13+` or `2.0.0`

## Installation

You can install XQuant with Python's [pip](https://pip.pypa.io/en/stable/) package manager:

```shell
pip install xquant
```
