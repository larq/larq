Once you have defined a good architecture for your BNN, you want to train it. Here we give an introduction to common training strategies and tricks that are popular in the field. After reading this guide you should have a good idea of how you can train your BNN.

Below we first discuss the fundamental challenges of BNN optimization. We then explain the most common used training strategy, of using latent weights, and cover the questions, tips & tricks that tend to come up when training BNNs.

## The Problem with SGD in BNNs

Stochastic Gradient Descent (SGD) is used pretty much everywhere in the field Deep Learning nowadays - either in its vanilla form or as the core part of some more sophisticated algorithm like [Adam](https://arxiv.org/abs/1412.6980).

However, when turning to BNNs, two fundamental issues arise with SGD:

- The gradient of the binarization operation is zero almost everywhere, making the gradient $\frac{\partial L}{\partial w}$ utterly uninformative.
- SGD performs optimization through small update steps that are accumulated over time. Binary weights, meanwhile, cannot absorb small updates: they can only be left alone, or flipped.

Another way of putting this is that the loss landscape for BNN is very different than what you are used to for real-valued networks. Gone are the glowing hills you can simply glide down from: the loss is now a discrete function, and many of the intuitions and theories developed for continuous loss landscapes no longer apply.

Luckily, there has been significant progress in solving these problems. The issue of zero gradients is resolved by replacing the gradient by some more informative 
alternative, what we call a 'pseudo-gradient'. The issue of updating can be resolved either by introducing latent weights, or by opting for a custom BNN optimizer.

## Latent Weights

Suppose we take a batch of training samples and evaluate a forward and backward pass. During the backward pass we replace the gradients with a pseudo-gradient, and we get a gradient vector on our weights. We then feed this into an optimizer like Adam, and get a vector with updates for our weights.

At this point, what do we do? If we directly apply the updates to our weights, they are no longer binary. The standard solution to this problem has been to introduce real-valued **latent weights**. We apply our update step to this real-valued weight. During the forward pass, we use the binarized version of the latent weight.

Beware that latent weights are [not really weights at all](https://arxiv.org/abs/1906.02107) - instead, they are best thought of as a product between the weight and a positive inertia: the higher this inertia, the stronger the signal required to make the weight flip.

One implication of this is that the latent weights should be constrained: as an increase in inertia does not change the behavior of the network, it can otherwise grow indefinitely.

In Larq, it is trivial to implement this strategy. An example of a layer optimized with this method would look like:

```python
x_out = larq.layers.QuantDense(512,
                               input_qunatizer="ste_sign",
                               kernel_quantizer="ste_sign",
                               kernel_constraint="weight_clip")(x_out)
```

Any optimizer you now apply will update the latent weights; after the update the latent weights are clipped to $[-1, 1]$.

### Alternative: Custom Optimizers

Instead of using latent weights, one can opt for a custom BNN optimizer that inherently generates binary weights. An example of such an optimizer is [Bop](api/optimizers/#bop).

## Choice of Pseudo-Gradient

In [`lq.quantizers`](/api/quantizers) you will find a variaty of quantizers that have been introduced in different papers. Many of these quantizers behave identically during the forward pass but implement different pseudo-gradients. Studies comparing different pseudo-gradients report little difference between them. Therefore, we recommend using the classical [`ste_sign()`](/api/quantizers/#ste_sign) as default.

## Choice of Optimizer

When using a latent weight strategy, you can apply any optimizer you are familair with from real-valued DL. However, due the different nature of BNNs your intuitions may be off. We recommend using Adam: although other optimizers can achieve similar accuracies with a lot of finetuning, we and others have found that Adam is easiest to use for most types of networks.

## Tips & Tricks

Here are some general tips and tricks that you may want to keep in mind:

- BNN training is more noisy due to non-continuous nature of flipping weights; therefore, we recommend setting your batch norm momentum to 0.9.
- Similarly, we recommend using a high batch size (e.g. 512, 1024 or even higher).
- Beware that BNNs tend to require many more epochs than real-valued networks to converge: 200+ epochs when training an Alexnet or Resnet-18 style network on ImageNet is not unusual.
- Networks tend to train much quicker if they are initialized from a trained real-valued model. Importantly, this requires the overal architecture of the pretrained network to be as similar as possible to the BNN, including placement of the activation operation (which replaces the binarization operation). Note that although convergence is faster, pretraining does not seem to improve final accuracy.

## Further References

If you would like to learn more, we recommend checking out the following papers (starting at the most recent):

- [Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization](https://arxiv.org/abs/1906.02107) - This paper investigates optimization of BNNs using latent weights and introduces [Bop](/api/optimizers/#bop) as the first custom BNN optimizer.
- [An Empirical study of Binary Neural Networks' Optimisation](https://openreview.net/forum?id=rJfUCoR5KX) - An emperical comparison of BNN optimization methods, including a detailed discussion on the use of various optimizers and a number of tricks used in the literature.