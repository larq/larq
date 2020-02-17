## Are there any restrictions on using Larq?

We believe open research and software is the way forward for the field of Binarized Neural Network (BNN) and deep learning and general: machine learning can be a powerful engine for good, but only if it is transparent, safe and widely accessible. That is why we have made Larq open-source and free to use.

Larq is licensed under [Apache 2.0 licence](https://github.com/larq/larq/blob/master/LICENSE) - you are free to contribute to [Larq](https://github.com/larq/larq), fork it or build commercial applications on top of it. If you distribute a modified version of Larq, you should clearly state changes you have made and consider contributing them back.

## What is a Binarized Neural Network (BNN)?

A BNN is a deep neural network in which the bulk of the computations are performed using binary values. For example, in a convolutional layer we can use weights that are either -1 or 1, and similarly binarize the activations.

Binarization enables the creation of deep learning models that are extremely efficient: storing the model only requires a single bit per weight, and evaluating the model can be done very efficiently because of the bitwise nature of the operations.

Note that in BNN not everything is binary: usually higher-precision computations are still used for things like the first layer, batch-normalization and residual connections.

## How can I cite Larq?

If Larq helps you in your work or research, it would be great if you can cite it as follows:

```bibtex
@article{larq,
  title     = {Larq: An Open-Source Library for Training Binarized Neural Networks},
  author    = {Lukas Geiger and Plumerai Team},
  year      = 2020,
  month     = jan,
  journal   = {Journal of Open Source Software},
  publisher = {The Open Journal},
  volume    = 5,
  number    = 45,
  pages     = 1746,
  doi       = {10.21105/joss.01746},
  url       = {https://doi.org/10.21105/joss.01746}
}
```

If your paper is publicly available, feel free to also add it to the list of [Papers using Larq](https://github.com/larq/larq/edit/master/docs/papers.md).

## Can I add my algorithm or model to Larq?

Absolutely! If you have developed a new model or training method that you would like to share with the community, create a PR or get in touch with us. Make sure you check out the contribution guide. For entire models with pretrained weights [`larq-zoo`](https://github.com/larq/zoo) is the correct place, everything else can be added to [`larq`](https://github.com/larq/larq) directly.

## Can I use Larq only for Binarized Neural Networks (BNNs)?

No, Larq is not just for BNNs! The real goal of Larq is to make it easy to work with extremely quantized networks. This includes BNNs as well as ternary networks (see for example [Ternary Weight Networks](https://arxiv.org/abs/1605.04711) or [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064)). Although the focus is currently on BNNs, Larq already supports a [ternary quantizer](/api/quantizers/#stetern) and binary and ternary networks have a lot in common. Moreover, modern BNNs are not 'pure' binary networks: they contain higher-precision first and last layers and shortcut connections, for example.

We will expand support for ternary and mixed-precision networks in the future by implementing layers and optimization tools that are useful in this context, and providing more examples. If there is any particular feature you would like to use, let us know by posting an [issue](https://github.com/larq/larq/issues).

## Why is Larq built on top of `tf.keras`?

We put a lot of thought into the question of which framework we should build Larq on and decided to go with TensorFlow / Keras over PyTorch or some other framework. There are a number of reasons for this:

- We really like the Keras API for its simplicity. At the same time, it is still very flexible if you want to build complex architectures or custom training loops.
- The TensorFlow ecosystem provides a wide range of tools for both researchers and developers. We think integration into that ecosystem will be beneficial for people working with BNNs.
- We are big fans of [`tf.datasets`](https://www.tensorflow.org/datasets/datasets).
- Reproducibility is a key concern to us, and our approach for [`larq-zoo`](https://github.com/larq/zoo) is heavily inspired by [Keras Applications](https://keras.io/applications/).

## Will there be a `PyTorch` version of Larq?

No, currently we are not planning on releasing a `PyTorch` version of Larq.

## Can I use Larq for the deployment of my models?

Currently, Larq is designed purely for training BNNs and there is no support for deployment of binarized models. This means that at this moment Larq is most useful to researchers and less so for developers working on applications.

Of course, the real goal of BNNs is efficient deployment, and in the future we will offer solutions for smooth deployment of models created and trained with Larq.
