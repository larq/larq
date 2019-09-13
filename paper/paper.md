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

# Summary

Bringing the power of deep learning outside of data centers can transform society: self-driving cars, mobile-based neural networks, and autonomous drones all have the potential to revolutionize everyday lives.
However, existing neural networks that use 32 bits, 16 bits or 8 bits to encode each weight and activation have an energy budget which is far beyond the scope of many of these applications.
Binarized Neural Networks (BNNs) have emerged as a promising solution to this problem. In these networks both weights and activations are restricted to $\{-1, +1\}$, resulting in models which are dramatically less computationally expensive, have a far lower memory footprint, and when executed on specialized hardware yield a stunning reduction in energy consumption [@Courbariaux2016].
However, BNNs are very difficult to design and train, so their use in industry is not yet widespread. `larq` is the first step towards solving this problem.

`larq` is an ecosystem of Python packages for BNNs and other Quantized Neural Networks (QNNs).
The API of `larq` is built on top of `tf.keras` [@tensorflow2015-whitepaper; @chollet2015keras] and is designed to provide an easy to use, composable way to design and train BNNs. It provides tools specifically designed to aid in BNN development, such as specialized optimizers, training metrics, and profiling tools.
It provides well-tested implementations and pretrained weights for a variety of popular extremely quantized models. This makes the field more accessible, encourages reproducibility, and facilitates research.

`larq` is built around the concept of quantized layers that compute

$$
\sigma(f(q_{\, \mathrm{kernel}}(\boldsymbol{w}), q_{\, \mathrm{input}}(\boldsymbol{x})) + b)
$$

with full precision weights $\boldsymbol{w}$, arbitrary precision input $\boldsymbol{x}$, layer operation $f$ (e.g. $f(\boldsymbol{w}, \boldsymbol{x}) = \boldsymbol{x}^T \boldsymbol{w}$ for a densely-connected layer), activation $\sigma$ and bias $b$.
$q_{\, \mathrm{kernel}}$ and $q_{\, \mathrm{input}}$ are quantizers that define an operation for quantizing a kernel and inputs, respectively, and a pseudo-gradient used for automatic differentiation.

The flexible yet easy to use API of `larq` is aimed at both researchers in the field of efficient deep learning [@2019arXiv190602107H] and practitioners who want to explore BNNs for their applications. Furthermore, `larq` makes it easier for beginners and students to get started with BNNs.
We are working to expand the audience by adding support for deploying BNNs on embedded devices, making `larq` useful for real applications. By building a community-driven open source project, we hope to accelerate research in the field of BNNs and other QNNs to enable deep learning in resource-constrained environments.

# References
