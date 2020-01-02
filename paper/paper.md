---
title: "Larq: An Open-Source Library for Training Binarized Neural Networks"
tags:
  - python
  - tensorflow
  - keras
  - deep-learning
  - machine-learning
  - binarized-neural-networks
  - quantized-neural-networks
  - efficient-deep-learning
authors:
  - name: Lukas Geiger
    orcid: 0000-0002-8697-9920
    affiliation: 1
  - name: Plumerai Team
    affiliation: 1
affiliations:
  - name: Plumerai Research
    index: 1
date: 9 September 2019
bibliography: paper.bib
---

# Introduction

Modern deep learning methods have been successfully applied to many different tasks and have the potential to revolutionize everyday lives. However, existing neural networks that use 32 bits to encode each weight and activation often have an energy budget that far exceeds the capabilities of mobile or embedded devices. One common way to improve computational efficiency is to reduce the precision of the network to 16-bit or 8-bit, also known as quantization.
Binarized Neural Networks (BNNs) represent an extreme case of quantized networks, that cannot be viewed as approximations to real-valued networks and therefore requires special tools and optimization strategies [@bop]. In these networks both weights and activations are restricted to $\{-1, +1\}$ [@binarynet]. Compared to an equivalent 8-bit quantized network BNNs require 8 times smaller memory size and 8 times fewer memory accesses, which reduces energy consumption drastically when deployed on optimized hardware [@binarynet].
However, many open research questions remain until the use of BNNs and other extremely quantized neural networks becomes widespread in industry. [`larq`](https://larq.dev) is an ecosystem of Python packages for BNNs and other Quantized Neural Networks (QNNs). It is intended to facilitate researchers to resolve these outstanding questions.

# Background: Neural Network Binarization

Binarization of artificial deep neural networks poses two fundamental challenges to regular deep learning strategies, which are all based on backpropagation and stochastic gradient descent [@rumelhart; @bottou1991]: the gradient of the binarization function vanishes almost everywhere and a weight in the set $\{-1, +1\}$ cannot absorb small update steps. Therefore, binarization was long deemed unfeasible.

A solution to both problems was suggested by Hinton, who proposed the Straight-Through Estimator, which ignores the binarization during the backward pass [@hinton_coursera]. The viability of this approach was demonstrated by BinaryConnect [@binaryconnect] and BinaryNet [@binarynet], proving that binarization of networks was possible even for complex tasks.

Since then, the field of BNNs and closely related Ternary Neural Networks has become a prime candidate to enable efficient inference for deep neural networks. Numerous papers have explored novel architectures [@Zhu2018; @bireal_net; @xnor_net; @Zhuang2018] and optimization strategies [@Alizadeh2019; @bop], and the accuracy gap between efficient BNNs and regular DNNs is rapidly closing.

# Training extremely quantized neural networks with `larq`

Existing tools for BNN research either focus on deployment or require a high learning curve to use and contribute to [@dabnn; @bmxnetv2]. The API of `larq` is built on top of `tensorflow.keras` [@tensorflow; @keras] and is designed to provide an easy to use, composable way to design and train BNNs. While popular libraries like TensorFlow Lite, TensorFlow Model Optimization or PyTorch focus on 16-bit or 8-bit quantization [@tensorflow; @pytorch], `larq` aims to extend this towards lower bit-widths.

Quantization is the process of mapping a set of continuous values to a smaller countable set. BNNs are a special case of QNNs, where the quantization output $x_q$ is binary:

$$
x_q = q(x), \quad x_q \in \{-1, +1\}, x \in \mathbb{R}
$$

`larq` is built around the concept of quantizers $q(x)$ and quantized layers that compute

$$
\sigma(f(q_{\, \mathrm{kernel}}(\boldsymbol{w}), q_{\, \mathrm{input}}(\boldsymbol{x})) + b)
$$

with full precision weights $\boldsymbol{w}$, arbitrary precision input $\boldsymbol{x}$, layer operation $f$ (e.g. $f(\boldsymbol{w}, \boldsymbol{x}) = \boldsymbol{x}^T \boldsymbol{w}$ for a densely-connected layer), activation $\sigma$ and bias $b$.
$q_{\, \mathrm{kernel}}$ and $q_{\, \mathrm{input}}$ are quantizers that define an operation for quantizing a kernel and inputs, respectively, and a pseudo-gradient used for automatic differentiation so that operations of the layer can be executed in reduced precision. The source code and documentation be found at [github.com/larq/larq](https://github.com/larq/larq) and [larq.dev](https://larq.dev) respectively.

While `larq` can be used to train networks with arbitrary bit-widths, it provides tools specifically designed to aid in BNN development, such as specialized optimizers, training metrics, and profiling tools.

# Encouraging reproducible research with `larq/zoo`

Fortunately, many researchers already publish code along with their papers, though, in absence of a common API to define extremely quantized networks, authors end up re-implementing a large amount of code, making it difficult to share improvements and make rapid progress. We and many other researchers in the field often encounter major problems when it comes to reproducing existing literature, due to incomplete or even broken code of official implementations.
To tackle this issue, [`larq/zoo`](https://larq.dev/models) provides tested and maintained implementations and pretrained weights for a variety of popular extremely quantized models [@binarynet; @bireal_net; @binary_dense_net; @xnor_net; @dorefa] helping researchers focus on their work instead of spending time on reproducing existing work.

# Summary

The flexible yet easy to use API of `larq` is aimed at both researchers in the field of efficient deep learning and practitioners who want to explore BNNs for their applications. Furthermore, `larq` makes it easier for beginners and students to get started with BNNs.
We are working to expand the audience by adding support for deploying BNNs on embedded devices, making `larq` useful for real applications. By building a community-driven open source project, we hope to accelerate research in the field of BNNs and other QNNs to enable deep learning in resource-constrained environments.

# References
