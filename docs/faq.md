## How can I cite Larq?

If Larq helps you in your work or research, it would be great if you can cite it as follows:

```
@misc{larq,
  author       = {Geiger, Lukas and Widdicombe, James and Bakhtiari, Arash and Helwegen, Koen and Heuss, Maria and Nusselder, Roeland},
  title        = {Larq: An Open-Source Deep Learning Library for Training Binarized Neural Networks},
  howpublished = {Web page},
  url          = {https://larq.dev},
  year         = {2019}
}
```

## Can I add my own model to Larq-Zoo?

Absolutely! If you have developed a new BNN that you would like to share with the community, create a PR or get in touch with us. Make sure you check out the contribution guide.

## Why is Larq built on top of `tf.keras`?

We put a lot of thought into the question of which framework we should build Larq on and decided to go with TensorFlow / Keras over PyTorch or some other framework. There are a number of reasons for this:

* We really like the Keras API for its simplicity. At the same time, it is still very flexible if you want to build complex architectures or custom training loops.
* The TensorFlow ecosystem provides a wide range of tools for both researchers and developers. We think integration into that ecosystem will be beneficial for people working with BNNs.
* We are big fans of [`tf.datasets`](https://www.tensorflow.org/datasets/datasets).
* Reproducibility is a key concern to us, and our approach for `larq-zoo` is heavily inspired by [Keras Applications](https://keras.io/applications/).

## Will there be a `PyTorch` version of Larq?

No, currently we are not planning onreleasing a `PyTorch` version of Larq.

## Can I use Larq for deployment of my models?

Currently Larq is designed purely for training BNNs and there is no support for deployment of binarized models. This means that at this moment Larq is most useful to researchers, and less so for developers working on applications.

Of course, the real goal of BNNs is efficient deployment, and in the future we will offer solutions for smooth deployment of models created and traind with Larq.