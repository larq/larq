# User Guide

If you're new to Larq and/or BNNs, this is the right place to start. Below we summarize the key concepts you need to understand to work with BNNs.

## Quantizer

The core idea of BNNs is to use binary values instead of 32-bit floating point. A [quantizer](https://larq.dev/api/quantizers/) defines the way of transforming a full precision input to a quantized output and the pseudo-gradient method used for the backwards pass.

Note that most layers in a neural network evaluate sums and thus output integers even if all inputs are binary. Therefore you will usually want to apply quantizers for your activations throughout the network even during inference.

It is also common to apply quantizers to the weights during training. This is necessary when relying on real-valued latent-weights to accumulate non-binary update steps, a common optimization strategy for BNNs. After training is finished, the real-valued weights and associated quantization operations can be discarded.

### Pseudo-Gradient

The true gradient of a quantizer is in general zero almost everywhere and therefore cannot be used for SGD. Instead, optimization of BNN relies on what we call pseudo-gradients, which are used during back-propogation. In the documentation for each quantizer you will find the definition and a graph of the pseudo-gradient.

## Quantized Layers

Each [quantized layers](https://larq.dev/api/layers/) requires an `input_quantizer` and a `kernel_quantizer` that describe the way of quantizing the incoming activations and weights of the layer respectively. If both `input_quantizer` and `kernel_quantizer` are `None` the layer is equivalent to a full precision layer.

A quantized layer computes

\\[
\sigma(f(q_{\, \mathrm{kernel}}(\boldsymbol{w}), q_{\, \mathrm{input}}(\boldsymbol{x})) + b)
\\]

with full precision weights \(\boldsymbol{w}\), arbitrary precision input \(\boldsymbol{x}\), layer operation \(f\) (e.g. \(f(\boldsymbol{w}, \boldsymbol{x}) = \boldsymbol{x}^T \boldsymbol{w}\) for a densely-connected layer), activation \(\sigma\) and bias \(b\). This will result in the following computational graph:

<div style="text-align:center;">
<svg width="50%" viewBox="0 0 249 238" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect width="249" height="238" fill="white"/>
<rect x="151" y="6" width="67" height="22" fill="#3F51B5"/>
<text fill="white" xml:space="preserve" style="white-space: pre" font-family="Roboto Mono" font-size="12" letter-spacing="0em"><tspan x="162.27" y="21.1016">kernel</tspan></text>
<rect x="162" y="127" width="67" height="22" fill="#3F51B5"/>
<text fill="white" xml:space="preserve" style="white-space: pre" font-family="Roboto Mono" font-size="12" letter-spacing="0em"><tspan x="180.477" y="142.102">bias</tspan></text>
<rect x="28" y="7" width="67" height="21" fill="#2196F3"/>
<text fill="white" xml:space="preserve" style="white-space: pre" font-family="Roboto Mono" font-size="12" letter-spacing="0em"><tspan x="43.4824" y="21.1016">input</tspan></text>
<rect x="90" y="209" width="67" height="21" fill="#2196F3"/>
<text fill="white" xml:space="preserve" style="white-space: pre" font-family="Roboto Mono" font-size="12" letter-spacing="0em"><tspan x="101.879" y="223.102">output</tspan></text>
<text fill="black" fill-opacity="0.54" xml:space="preserve" style="white-space: pre" font-family="Roboto Mono" font-size="12" letter-spacing="0em"><tspan x="7.44727" y="64.1016">input_quantizer</tspan></text>
<text fill="black" fill-opacity="0.54" xml:space="preserve" style="white-space: pre" font-family="Roboto Mono" font-size="12" letter-spacing="0em"><tspan x="126.844" y="64.1016">kernel_quantizer</tspan></text>
<text fill="black" fill-opacity="0.54" xml:space="preserve" style="white-space: pre" font-family="Roboto Mono" font-size="12" letter-spacing="0em"><tspan x="69.4473" y="103.102">layer_operation</tspan></text>
<text fill="black" fill-opacity="0.54" xml:space="preserve" style="white-space: pre" font-family="Roboto Mono" font-size="12" letter-spacing="0em"><tspan x="112.689" y="142.102">add</tspan></text>
<text fill="black" fill-opacity="0.54" xml:space="preserve" style="white-space: pre" font-family="Roboto Mono" font-size="12" letter-spacing="0em"><tspan x="87.4648" y="181.102">activation</tspan></text>
<path d="M60.6464 48.3536C60.8417 48.5488 61.1583 48.5488 61.3536 48.3536L64.5355 45.1716C64.7308 44.9763 64.7308 44.6597 64.5355 44.4645C64.3403 44.2692 64.0237 44.2692 63.8284 44.4645L61 47.2929L58.1716 44.4645C57.9763 44.2692 57.6597 44.2692 57.4645 44.4645C57.2692 44.6597 57.2692 44.9763 57.4645 45.1716L60.6464 48.3536ZM60.5 32V48H61.5V32H60.5Z" fill="black" fill-opacity="0.54"/>
<path d="M183.646 48.3536C183.842 48.5488 184.158 48.5488 184.354 48.3536L187.536 45.1716C187.731 44.9763 187.731 44.6597 187.536 44.4645C187.34 44.2692 187.024 44.2692 186.828 44.4645L184 47.2929L181.172 44.4645C180.976 44.2692 180.66 44.2692 180.464 44.4645C180.269 44.6597 180.269 44.9763 180.464 45.1716L183.646 48.3536ZM183.5 32V48H184.5V32H183.5Z" fill="black" fill-opacity="0.54"/>
<path d="M123.646 204.354C123.842 204.549 124.158 204.549 124.354 204.354L127.536 201.172C127.731 200.976 127.731 200.66 127.536 200.464C127.34 200.269 127.024 200.269 126.828 200.464L124 203.293L121.172 200.464C120.976 200.269 120.66 200.269 120.464 200.464C120.269 200.66 120.269 200.976 120.464 201.172L123.646 204.354ZM123.5 188V204H124.5V188H123.5Z" fill="black" fill-opacity="0.54"/>
<path d="M123.646 165.354C123.842 165.549 124.158 165.549 124.354 165.354L127.536 162.172C127.731 161.976 127.731 161.66 127.536 161.464C127.34 161.269 127.024 161.269 126.828 161.464L124 164.293L121.172 161.464C120.976 161.269 120.66 161.269 120.464 161.464C120.269 161.66 120.269 161.976 120.464 162.172L123.646 165.354ZM123.5 149V165H124.5V149H123.5Z" fill="black" fill-opacity="0.54"/>
<path d="M123.646 126.354C123.842 126.549 124.158 126.549 124.354 126.354L127.536 123.172C127.731 122.976 127.731 122.66 127.536 122.464C127.34 122.269 127.024 122.269 126.828 122.464L124 125.293L121.172 122.464C120.976 122.269 120.66 122.269 120.464 122.464C120.269 122.66 120.269 122.976 120.464 123.172L123.646 126.354ZM123.5 110V126H124.5V110H123.5Z" fill="black" fill-opacity="0.54"/>
<path d="M140.624 137.647C140.441 137.842 140.461 138.158 140.669 138.353L144.049 141.529C144.256 141.724 144.573 141.724 144.756 141.529C144.939 141.334 144.919 141.018 144.712 140.823L141.707 138L144.359 135.177C144.542 134.982 144.522 134.666 144.315 134.471C144.108 134.276 143.791 134.276 143.608 134.471L140.624 137.647ZM157 137.501L140.969 137.501L141.031 138.499L157.062 138.499L157 137.501Z" fill="black" fill-opacity="0.54"/>
<path d="M150.5 85.3137C150.5 85.5899 150.724 85.8137 151 85.8137H155.5C155.776 85.8137 156 85.5899 156 85.3137C156 85.0376 155.776 84.8137 155.5 84.8137H151.5V80.8137C151.5 80.5376 151.276 80.3137 151 80.3137C150.724 80.3137 150.5 80.5376 150.5 80.8137L150.5 85.3137ZM161.96 73.6464L150.646 84.9602L151.354 85.6673L162.667 74.3536L161.96 73.6464Z" fill="black" fill-opacity="0.54"/>
<path d="M97.3137 85.8137C97.5899 85.8137 97.8137 85.5899 97.8137 85.3137V80.8137C97.8137 80.5376 97.5899 80.3137 97.3137 80.3137C97.0376 80.3137 96.8137 80.5376 96.8137 80.8137V84.8137H92.8137C92.5376 84.8137 92.3137 85.0376 92.3137 85.3137C92.3137 85.5899 92.5376 85.8137 92.8137 85.8137H97.3137ZM85.6464 74.3536L96.9602 85.6673L97.6673 84.9602L86.3536 73.6464L85.6464 74.3536Z" fill="black" fill-opacity="0.54"/>
</svg>
</div>

Larq layers are fully compatible with the Keras API so you can use them with Keras Layers interchangeably:

```python tab="Larq 32-bit model"
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    larq.layers.QuantDense(512, activation="relu"),
    larq.layers.QuantDense(10, activation="softmax")
])
```

```python tab="Keras 32-bit model"
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

A simple fully-connected Binarized Neural Network (BNN) using the [Straight-Through Estimator](https://larq.dev/api/quantizers/#ste_sign) can be defined in just a few lines of code using either the Keras sequential, functional or model subclassing APIs:

```python tab="Larq 1-bit model"
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

```python tab="Larq 1-bit model functional"
x = tf.keras.Input(shape=(28, 28, 1))
y = tf.keras.layers.Flatten()(x)
y = larq.layers.QuantDense(512,
                           kernel_quantizer="ste_sign",
                           kernel_constraint="weight_clip")(y)
y = larq.layers.QuantDense(10,
                           input_quantizer="ste_sign",
                           kernel_quantizer="ste_sign",
                           kernel_constraint="weight_clip",
                           activation="softmax")(y)
model = tf.keras.Model(inputs=x, outputs=y)
```

```python tab="Larq 1-bit model subclassing"
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = larq.layers.QuantDense(512,
                                             kernel_quantizer="ste_sign",
                                             kernel_constraint="weight_clip")
        self.dense2 = larq.layers.QuantDense(10,
                                             input_quantizer="ste_sign",
                                             kernel_quantizer="ste_sign",
                                             kernel_constraint="weight_clip",
                                             activation="softmax")

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)

model = MyModel()
```

## Using Custom Quantizers

Quantizers are functions that transform a full precision input to a quantized output. Since this transformation usually is non-differentiable it is necessary to modify the gradient in order to be able to train the resulting QNN. This can be done with the [`tf.custom_gradient`](https://www.tensorflow.org/api_docs/python/tf/custom_gradient) decorator.

In this example we will define a binarization function with an identity gradient:

```python
@tf.custom_gradient
def identity_sign(x):
    def grad(dy):
        return dy
    return tf.sign(x), grad
```

This function can now be used as an `input_quantizer` or a `kernel_quantizer`:

```python
larq.layers.QuantDense(10,
                       input_quantizer=identity_sign,
                       kernel_quantizer=identity_sign)
```
