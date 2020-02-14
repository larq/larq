Here you will find a quick overview of the best practices that have evolved in the BNN community over the past few years regarding BNN architecture. After following this guide, you should be able to start designing a BNN for your application of interest that is both efficient and powerful.

## Binarizing a Single Layer

Any layer has two types of inputs: the layer parameters, such as a weight matrix and biases, and incoming activations.

We can reduce the memory footprint of the model by binarizing parameters. In Larq, this can be done by passing a `kernel_quantizer` from [`larq.quantizers`](/api/quantizers) when instantiating a [`larq.layer`](/api/layers) object, or by using a custom BNN optimizer such as [Bop](/api/optimizers/#bop) (note that by default Bop only targets kernels of layers in [`larq.layers`](/api/layers)).

To get the efficiency of binary computations, the incoming activations need to be binary as well. This can be done by setting a `input_quantizer`.

Note that the output of a binarized layer is _not_ binary. Instead the output is an integer, due to the summation that appears in most neural network layers.

When viewing binarization as an activation function just like ReLU, one may be inclined to binarize the outgoing activations rather than the incoming activations. However, if the network contains batch normalization layers or residual connections, this may result in unintentional non-binary operations. Therefore we have opted for an `input_quantizer` rather than an `activation_quantizer`.

A typical binarized layer looks something like:

```python
import larq as lq
...
x_out = lq.layers.QuantDense(
    units=1024,
    input_quantizer=lq.quantizers.SteSign(clip_value=1.0),
    kernel_quantizer=lq.quantizers.SteSign(clip_value=1.0),
    kernel_constraint=lq.constraints.WeightClip(clip_value=1.0),
    )(x_in)
```

## First & Last Layer

Binarizing the first and last layers hurts accuracy much more than binarizing other layers in the network. Meanwhile, the number of weights and operations in these layers are relatively small. Therefore it has become standard to leave these layers in higher precision. This applies to the incoming activations as well as the weights.
The following shows a network that was trained on CIFAR10 with different precisions for the first and last layers.

```plot-altair
/plots/first_and_last_layers.vg.json
```

## Batch Normalization

Perhaps somewhat surprisingly, batch normalization remains crucial in BNNs. All successful BNNs still contain a batch norm layer after each binarized layer. We recommend using batch normalization with trainable beta and gamma.

Note that when no residuals are used, the batch norm operation can be simplified (see e.g. Fig. 2 in [this paper](https://arxiv.org/pdf/1904.02823.pdf)).

## High-Precision Shortcuts

A binarized layer outputs an integer activation matrix that is binarized before the next layer. This means that in a VGG-style network such as [BinaryNet](https://arxiv.org/abs/1602.02830) information is lost between every two layers, and one may wonder if this is optimal in terms of efficiency.

High-precision shortcuts avoid this loss of information. Examples of networks that include such shortcuts are [Bi-Real net](https://arxiv.org/abs/1808.00278) and [Binary Densenets](https://arxiv.org/abs/1906.08637). Note that the argument for introducing these shortcuts is no longer just to improve gradient flow, as it is in real-valued models: in BNNs, high-precision shortcuts really improve the expressivity of the model.

Such shortcuts are relatively cheap in terms of memory footprint and computational cost, and they greatly improve accuracy. Beware that they do increase the runtime memory requirements of the model.

An issue with ResNet-style shortcuts comes up when there is a dimensionality change. Currently, the most popular solution to this is to use pointwise high-precision convolutions in the residual connections if there is a dimensionality change.

We recommend to use as many shortcuts as you can: for example, in ResNet-style architectures it helps to bypass every single convolutional layer, instead of every two convolutional layers.

An example of a convolutional block with shortcut in Larq could look something like this:

```python
import larq as lq
...
def conv_with_shortcut(x):
    """Convolutional block with shortcut connection.

    Args:
      x: input tensor with high-precision activations

    Returns:
      Tensor with high-precision activations
    """
    # get number of filters
    filters = x.get_shape().as_list()[-1]

    # create shortcut that retains the high-precision information
    shortcut = x

    # efficient binarized convolutions (note inputs are also binarized)
    x = lq.layers.QuantConv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        input_quantizer=lq.quantizers.SteSign(clip_value=1.0),
        kernel_quantizer=lq.quantizers.SteSign(clip_value=1.0),
        kernel_initializer="glorot_normal",
        kernel_constraint=lq.constraints.WeightClip(clip_value=1.0),
        use_bias=False,
    )(x)

    # normalize the (integer) output of the binary conv and merge
    # with shortcut
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    out = tf.keras.layers.add([x, shortcut])

    return out
```

## Pooling

The [XNOR-net authors](https://arxiv.org/abs/1603.05279) found that accuracy improves when applying batch normalization after instead of before max-pooling. In general, max pooling in BNNs can be problematic as it can lead to skewed binarized activations.

Thus, in a VGG-style network a layer could look like this:

```python
import larq as lq
...
x = lq.layers.QuantConv2D(
    filters=512,
    kernel_size=3,
    padding="same",
    input_quantizer=lq.quantizers.SteSign(clip_value=1.0),
    kernel_quantizer=lq.quantizers.SteSign(clip_value=1.0),
    kernel_constraint=lq.constraints.WeightClip(clip_value=1.0),
    use_bias=False,
    )(x)
x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
x = tf.keras.layers.BatchNormalization(momentum=.9)(x)
```

## Further References

If you would like to learn more, we recommend checking out the following papers (starting at the most recent):

- [Back to Simplicity: How to Train Accurate BNNs from Scratch?](https://arxiv.org/abs/1906.08637) - This recent paper introduces Binary Densenets, demonstrating good results on ImageNet. The authors take an information-theoretic perspective on BNN architectures and give a number of recommendations for good architecture design.
- [Structured Binary Neural Networks for Accurate Image Classification and Semantic Segmentation](https://arxiv.org/abs/1811.10413) - A thought-provoking paper that presents Group-Net. The authors question whether architectural features developed for real-valued networks are the most appropriate for BNN. Even more than the presented architecture, we find this line of thinking very interesting and hope Larq will enable people to explore novel ideas more easily.
- [Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm](https://arxiv.org/abs/1808.00278) - This ECCV 2018 paper introduces Bi-Real nets, one of the first binarized networks that use high-precision shortcuts.
- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830) - The classic BNN paper, mandatory reading for anyone working in the field. In addition to introducing many of the foundational ideas for BNNs, the paper contains an interesting discussion on batch normalization.
