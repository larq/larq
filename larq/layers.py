"""Each Quantized Layer requires a `input_quantizer` and `kernel_quantizer` that
describes the way of quantizing the activation of the previous layer and the weights
respectively.

If both `input_quantizer` and `kernel_quantizer` are `None` the layer
is equivalent to a full precision layer.
"""

import tensorflow as tf

from larq import utils
from larq.layers_base import (
    QuantizerBase,
    QuantizerBaseConv,
    QuantizerDepthwiseBase,
    QuantizerSeparableBase,
)


@utils.register_keras_custom_object
class QuantDense(QuantizerBase, tf.keras.layers.Dense):
    """Just your regular densely-connected quantized NN layer.

    `QuantDense` implements the operation:
    `output = activation(dot(input_quantizer(input), kernel_quantizer(kernel)) + bias)`,
    where `activation` is the element-wise activation function passed as the
    `activation` argument, `kernel` is a weights matrix created by the layer, and `bias`
    is a bias vector created by the layer (only applicable if `use_bias` is `True`).
    `input_quantizer` and `kernel_quantizer` are the element-wise quantization
    functions to use. If both quantization functions are `None` this layer is
    equivalent to `Dense`.

    !!! note ""
        If the input to the layer has a rank greater than 2, then it is flattened
        prior to the initial dot product with `kernel`.

    !!! example
        ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(
            QuantDense(
                32,
                input_quantizer="ste_sign",
                kernel_quantizer="ste_sign",
                kernel_constraint="weight_clip",
                input_shape=(16,),
            )
        )
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(
            QuantDense(
                32,
                input_quantizer="ste_sign",
                kernel_quantizer="ste_sign",
                kernel_constraint="weight_clip",
            )
        )
        ```

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use. If you don't specify anything,
            no activation is applied (`a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        input_quantizer: Quantization function applied to the input of the layer.
        kernel_quantizer: Quantization function applied to the `kernel` weights matrix.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.

    # Input shape
        N-D tensor with shape: `(batch_size, ..., input_dim)`. The most common situation
        would be a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        N-D tensor with shape: `(batch_size, ..., units)`. For instance, for a 2D input
        with shape `(batch_size, input_dim)`, the output would have shape
        `(batch_size, units)`.
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


@utils.register_keras_custom_object
class QuantConv1D(QuantizerBase, QuantizerBaseConv, tf.keras.layers.Conv1D):
    """1D quantized convolution layer (e.g. temporal convolution).

    This layer creates a convolution kernel that is convolved with the layer input
    over a single spatial (or temporal) dimension to produce a tensor of outputs.
    `input_quantizer` and `kernel_quantizer` are the element-wise quantization
    functions to use. If both quantization functions are `None` this layer is
    equivalent to `Conv1D`.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide an `input_shape`
    argument (tuple of integers or `None`, e.g. `(10, 128)` for sequences of
    10 vectors of 128-dimensional vectors, or `(None, 128)` for variable-length
    sequences of 128-dimensional vectors.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the stride
            length of the convolution. Specifying any stride value != 1 is incompatible
            with specifying any `dilation_rate` value != 1.
        padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive). `"causal"`
            results in causal (dilated) convolutions, e.g. output[t] does not depend on
            input[t+1:]. Useful when modeling temporal data where the model should not
            violate the temporal order. See [WaveNet: A Generative Model for Raw Audio,
                section 2.1](https://arxiv.org/abs/1609.03499).
        pad_values: The pad value to use when `padding="same"`.
        data_format: A string, one of `channels_last` (default) or `channels_first`.
        dilation_rate: an integer or tuple/list of a single integer, specifying the
            dilation rate to use for dilated convolution. Currently, specifying any
            `dilation_rate` value != 1 is incompatible with specifying any `strides`
            value != 1.
        activation: Activation function to use. If you don't specify anything, no
            activation is applied (`a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        input_quantizer: Quantization function applied to the input of the layer.
        kernel_quantizer: Quantization function applied to the `kernel` weights matrix.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.

    # Input shape
        3D tensor with shape: `(batch_size, steps, input_dim)`

    # Output shape
        3D tensor with shape: `(batch_size, new_steps, filters)`.
        `steps` value might have changed due to padding or strides.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        pad_values=0.0,
        data_format="channels_last",
        dilation_rate=1,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            pad_values=pad_values,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


@utils.register_keras_custom_object
class QuantConv2D(QuantizerBase, QuantizerBaseConv, tf.keras.layers.Conv2D):
    """2D quantized convolution layer (e.g. spatial convolution over images).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    `input_quantizer` and `kernel_quantizer` are the element-wise quantization
    functions to use. If both quantization functions are `None` this layer is
    equivalent to `Conv2D`. If `use_bias` is True, a bias vector is created
    and added to the outputs. Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword argument
    `input_shape` (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures in
    `data_format="channels_last"`.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window. Can be a single integer
            to specify the same value for all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides of
            the convolution along the height and width. Can be a single integer to
            specify the same value for all spatial dimensions. Specifying any stride
            value != 1 is incompatible with specifying any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        pad_values: The pad value to use when `padding="same"`.
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs. `channels_last` corresponds to
            inputs with shape `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape `(batch, channels, height, width)`. It
            defaults to the `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution. Can be a single integer to specify the
            same value for all spatial dimensions. Currently, specifying any
            `dilation_rate` value != 1 is incompatible with specifying any stride value
            != 1.
        activation: Activation function to use. If you don't specify anything,
            no activation is applied (`a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        input_quantizer: Quantization function applied to the input of the layer.
        kernel_quantizer: Quantization function applied to the `kernel` weights matrix.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        pad_values=0.0,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            pad_values=pad_values,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


@utils.register_keras_custom_object
class QuantConv3D(QuantizerBase, QuantizerBaseConv, tf.keras.layers.Conv3D):
    """3D convolution layer (e.g. spatial convolution over volumes).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. `input_quantizer` and `kernel_quantizer` are the element-wise quantization
    functions to use. If both quantization functions are `None` this layer is
    equivalent to `Conv3D`. If `use_bias` is True, a bias vector is created and
    added to the outputs. Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword argument
    `input_shape` (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
    with a single channel, in `data_format="channels_last"`.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, height and width of the 3D convolution window. Can be a single
            integer to specify the same value for all spatial dimensions.
        strides: An integer or tuple/list of 3 integers, specifying the strides of the
            convolution along each spatial dimension. Can be a single integer to specify
            the same value for all spatial dimensions. Specifying any stride value != 1
            is incompatible with specifying any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        pad_values: The pad value to use when `padding="same"`.
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs. `channels_last` corresponds to
            inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while
            `channels_first` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`. It defaults
            to the `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution. Can be a single integer to specify the
            same value for all spatial dimensions. Currently, specifying any
            `dilation_rate` value != 1 is incompatible with specifying any stride value
            != 1.
        activation: Activation function to use. If you don't specify anything,
            no activation is applied (`a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        input_quantizer: Quantization function applied to the input of the layer.
        kernel_quantizer: Quantization function applied to the `kernel` weights matrix.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.

    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if
            data_format='channels_first'
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if
            data_format='channels_last'.

    # Output shape
        5D tensor with shape:
        `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if
            data_format='channels_first'
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if
            data_format='channels_last'.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have
            changed due to padding.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding="valid",
        pad_values=0.0,
        data_format=None,
        dilation_rate=(1, 1, 1),
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            pad_values=pad_values,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


@utils.register_keras_custom_object
class QuantDepthwiseConv2D(
    QuantizerDepthwiseBase, QuantizerBaseConv, tf.keras.layers.DepthwiseConv2D
):
    """Quantized depthwise separable 2D convolution.

    Depthwise Separable convolutions consists in performing just the first step in a
    depthwise spatial convolution (which acts on each input channel separately).
    The `depth_multiplier` argument controls how many output channels are generated per
    input channel in the depthwise step.

    # Arguments
        kernel_size: An integer or tuple/list of 2 integers, specifying the height and
            width of the 2D convolution window. Can be a single integer to specify the
            same value for all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides of the
            convolution along the height and width. Can be a single integer to specify
            the same value for all spatial dimensions. Specifying any stride value != 1
            is incompatible with specifying any `dilation_rate` value != 1.
        padding: one of `'valid'` or `'same'` (case-insensitive).
        pad_values: The pad value to use when `padding="same"`.
        depth_multiplier: The number of depthwise convolution output channels for each
            input channel. The total number of depthwise convolution output channels
            will be equal to `filters_in * depth_multiplier`.
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs. `channels_last` corresponds to
            inputs with shape `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be 'channels_last'.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied (ie. `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        input_quantizer: Quantization function applied to the input of the layer.
        depthwise_quantizer: Quantization function applied to the `depthwise_kernel`
            weights matrix.
        depthwise_initializer: Initializer for the depthwise kernel matrix.
        bias_initializer: Initializer for the bias vector.
        depthwise_regularizer: Regularizer function applied to the depthwise kernel
            matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its 'activation').
        depthwise_constraint: Constraint function applied to the depthwise kernel
            matrix.
        bias_constraint: Constraint function applied to the bias vector.

    # Input shape
        4D tensor with shape:
        `[batch, channels, rows, cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, rows, cols, channels]` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `[batch, filters, new_rows, new_cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, new_rows, new_cols, filters]` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        pad_values=0.0,
        depth_multiplier=1,
        data_format=None,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        depthwise_quantizer=None,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            pad_values=pad_values,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            depthwise_quantizer=depthwise_quantizer,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


@utils.register_keras_custom_object
class QuantSeparableConv1D(
    QuantizerSeparableBase, QuantizerBaseConv, tf.keras.layers.SeparableConv1D
):
    """Depthwise separable 1D quantized convolution.

    This layer performs a depthwise convolution that acts separately on channels,
    followed by a pointwise convolution that mixes channels.
    `input_quantizer`, `depthwise_quantizer` and `pointwise_quantizer` are the
    element-wise quantization functions to use. If all quantization functions are `None`
    this layer is equivalent to `SeparableConv1D`. If `use_bias` is True and
    a bias initializer is provided, it adds a bias vector to the output.
    It then optionally applies an activation function to produce the final output.

    # Arguments
        filters: Integer, the dimensionality of the output space (i.e. the number
            of filters in the convolution).
        kernel_size: A single integer specifying the spatial dimensions of the filters.
        strides: A single integer specifying the strides of the convolution.
            Specifying any `stride` value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"`, `"same"`, or `"causal"` (case-insensitive).
        pad_values: The pad value to use when `padding="same"`.
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs. `channels_last` corresponds
            to inputs with shape `(batch, length, channels)` while `channels_first`
            corresponds to inputs with shape `(batch, channels, length)`.
        dilation_rate: A single integer, specifying the dilation rate to use for dilated
            convolution. Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        depth_multiplier: The number of depthwise convolution output channels for
            each input channel. The total number of depthwise convolution output
            channels will be equal to `num_filters_in * depth_multiplier`.
        activation: Activation function. Set it to None to maintain a linear activation.
        use_bias: Boolean, whether the layer uses a bias.
        input_quantizer: Quantization function applied to the input of the layer.
        depthwise_quantizer: Quantization function applied to the depthwise kernel.
        pointwise_quantizer: Quantization function applied to the pointwise kernel.
        depthwise_initializer: An initializer for the depthwise convolution kernel.
        pointwise_initializer: An initializer for the pointwise convolution kernel.
        bias_initializer: An initializer for the bias vector. If None, the default
            initializer will be used.
        depthwise_regularizer: Optional regularizer for the depthwise convolution
            kernel.
        pointwise_regularizer: Optional regularizer for the pointwise convolution
            kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        depthwise_constraint: Optional projection function to be applied to the
            depthwise kernel after being updated by an `Optimizer`
            (e.g. used for norm constraints or value constraints for layer weights).
            The function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints are
            not safe to use when doing asynchronous distributed training.
        pointwise_constraint: Optional projection function to be applied to the
            pointwise kernel after being updated by an `Optimizer`.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        trainable: Boolean, if `True` the weights of this layer will be marked as
            trainable (and listed in `layer.trainable_weights`).
        name: A string, the name of the layer.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        pad_values=0.0,
        data_format=None,
        dilation_rate=1,
        depth_multiplier=1,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        depthwise_quantizer=None,
        pointwise_quantizer=None,
        depthwise_initializer="glorot_uniform",
        pointwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            pad_values=pad_values,
            data_format=data_format,
            dilation_rate=dilation_rate,
            depth_multiplier=depth_multiplier,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            depthwise_quantizer=depthwise_quantizer,
            pointwise_quantizer=pointwise_quantizer,
            depthwise_initializer=depthwise_initializer,
            pointwise_initializer=pointwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            pointwise_regularizer=pointwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            pointwise_constraint=pointwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


@utils.register_keras_custom_object
class QuantSeparableConv2D(
    QuantizerSeparableBase, QuantizerBaseConv, tf.keras.layers.SeparableConv2D
):
    """Depthwise separable 2D convolution.

    Separable convolutions consist in first performing a depthwise spatial convolution
    (which acts on each input channel separately) followed by a pointwise convolution
    which mixes together the resulting output channels. The `depth_multiplier` argument
    controls how many output channels are generated per input channel
    in the depthwise step.
    `input_quantizer`, `depthwise_quantizer` and `pointwise_quantizer` are the
    element-wise quantization functions to use. If all quantization functions are `None`
    this layer is equivalent to `SeparableConv1D`. If `use_bias` is True and
    a bias initializer is provided, it adds a bias vector to the output.
    It then optionally applies an activation function to produce the final output.

    Intuitively, separable convolutions can be understood as a way to factorize a
    convolution kernel into two smaller kernels,
    or as an extreme version of an Inception block.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the height and
            width of the 2D convolution window. Can be a single integer to specify the
            same value for all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides of the
            convolution along the height and width. Can be a single integer to specify
            the same value for all spatial dimensions. Specifying any stride value != 1
            is incompatible with specifying any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        pad_values: The pad value to use when `padding="same"`.
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs. `channels_last` corresponds to
            inputs with shape `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape `(batch, channels, height, width)`. It
            defaults to the `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution. Can be a single integer to specify the
            same value for all spatial dimensions. Currently, specifying any
            `dilation_rate` value != 1 is incompatible with specifying any stride value
            != 1.
        depth_multiplier: The number of depthwise convolution output channels for each
            input channel. The total number of depthwise convolution output channels
            will be equal to `filters_in * depth_multiplier`.
        activation: Activation function to use. If you don't specify anything,
            no activation is applied (`a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        input_quantizer: Quantization function applied to the input of the layer.
        depthwise_quantizer: Quantization function applied to the depthwise kernel
            matrix.
        pointwise_quantizer: Quantization function applied to the pointwise kernel
            matrix.
        depthwise_initializer: Initializer for the depthwise kernel matrix.
        pointwise_initializer: Initializer for the pointwise kernel matrix.
        bias_initializer: Initializer for the bias vector.
        depthwise_regularizer: Regularizer function applied to the depthwise kernel
            matrix.
        pointwise_regularizer: Regularizer function applied to the pointwise kernel
            matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        depthwise_constraint: Constraint function applied to the depthwise kernel
            matrix.
        pointwise_constraint: Constraint function applied to the pointwise kernel
            matrix.
        bias_constraint: Constraint function applied to the bias vector.`

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        pad_values=0.0,
        data_format=None,
        dilation_rate=(1, 1),
        depth_multiplier=1,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        depthwise_quantizer=None,
        pointwise_quantizer=None,
        depthwise_initializer="glorot_uniform",
        pointwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            pad_values=pad_values,
            data_format=data_format,
            dilation_rate=dilation_rate,
            depth_multiplier=depth_multiplier,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            depthwise_quantizer=depthwise_quantizer,
            pointwise_quantizer=pointwise_quantizer,
            depthwise_initializer=depthwise_initializer,
            pointwise_initializer=pointwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            pointwise_regularizer=pointwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            pointwise_constraint=pointwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


@utils.register_keras_custom_object
class QuantConv2DTranspose(QuantizerBase, tf.keras.layers.Conv2DTranspose):
    """Transposed quantized convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises from the desire to use a
    transformation going in the opposite direction of a normal convolution, i.e.,
    from something that has the shape of the output of some convolution to something
    that has the shape of its input while maintaining a connectivity pattern
    that is compatible with said convolution. `input_quantizer` and `kernel_quantizer`
    are the element-wise quantization functions to use. If both quantization functions
    are `None` this layer is equivalent to `Conv2DTranspose`.

    When using this layer as the first layer in a model, provide the keyword argument
    `input_shape` (tuple of integers, does not include the sample axis), e.g.
    `input_shape=(128, 128, 3)` for 128x128 RGB pictures in
    `data_format="channels_last"`.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window. Can be a single integer
            to specify the same value for all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides of
            the convolution along the height and width. Can be a single integer to
            specify the same value for all spatial dimensions. Specifying any stride
            value != 1 is incompatible with specifying any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        output_padding: An integer or tuple/list of 2 integers, specifying the amount
            of padding along the height and width of the output tensor. Can be a single
            integer to specify the same value for all spatial dimensions. The amount of
            output padding along a given dimension must be lower than the stride along
            that same dimension.
            If set to `None` (default), the output shape is inferred.
        data_format: A string, one of `channels_last` (default) or `channels_first`. The
            ordering of the dimensions in the inputs. `channels_last` corresponds to
            inputs with shape `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape `(batch, channels, height, width)`. It
            defaults to the `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution. Can be a single integer to specify the
            same value for all spatial dimensions. Currently, specifying any
            `dilation_rate` value != 1 is incompatible with specifying any stride value
            != 1.
        activation: Activation function to use. If you don't specify anything,
            no activation is applied (`a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        input_quantizer: Quantization function applied to the input of the layer.
        kernel_quantizer: Quantization function applied to the `kernel` weights matrix.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.

    # References
        - [A guide to convolution arithmetic for deep
            learning](https://arxiv.org/abs/1603.07285v1)
        - [Deconvolutional
            Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        output_padding=None,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


@utils.register_keras_custom_object
class QuantConv3DTranspose(QuantizerBase, tf.keras.layers.Conv3DTranspose):
    """Transposed quantized convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution. `input_quantizer` and `kernel_quantizer`
    are the element-wise quantization functions to use. If both quantization functions
    are `None` this layer is equivalent to `Conv3DTranspose`.

    When using this layer as the first layer in a model, provide the keyword argument
    `input_shape` (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 128, 3)` for a 128x128x128 volume with 3 channels
    if `data_format="channels_last"`.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 3 integers, specifying the depth, height
            and width of the 3D convolution window. Can be a single integer to specify the
            same value for all spatial dimensions.
        strides: An integer or tuple/list of 3 integers, specifying the strides of the
            convolution along the depth, height and width. Can be a single integer to
            specify the same value for all spatial dimensions. Specifying any stride
            value != 1 is incompatible with specifying any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        output_padding: An integer or tuple/list of 3 integers, specifying the amount
            of padding along the depth, height, and width. Can be a single integer to
            specify the same value for all spatial dimensions. The amount of output
            padding along a given dimension must be lower than the stride along that
            same dimension. If set to `None` (default), the output shape is inferred.
        data_format: A string, one of `channels_last` (default) or `channels_first`. The
            ordering of the dimensions in the inputs. `channels_last` corresponds to
            inputs with shape `(batch, depth, height, width, channels)` while
            `channels_first` corresponds to inputs with shape
            `(batch, channels, depth, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution. Can be a single integer to specify the
            same value for all spatial dimensions. Currently, specifying any
            `dilation_rate` value != 1 is incompatible with specifying any stride value
            != 1.
        activation: Activation function to use. If you don't specify anything,
            no activation is applied (`a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        input_quantizer: Quantization function applied to the input of the layer.
        kernel_quantizer: Quantization function applied to the `kernel` weights matrix.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.

    # Input shape
        5D tensor with shape:
        `(batch, channels, depth, rows, cols)` if data_format='channels_first'
        or 5D tensor with shape:
        `(batch, depth, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        5D tensor with shape:
        `(batch, filters, new_depth, new_rows, new_cols)` if data_format='channels_first'
        or 5D tensor with shape:
        `(batch, new_depth, new_rows, new_cols, filters)` if data_format='channels_last'.
        `depth` and `rows` and `cols` values might have changed due to padding.

    # References
        - [A guide to convolution arithmetic for deep
            learning](https://arxiv.org/abs/1603.07285v1)
        - [Deconvolutional
            Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding="valid",
        output_padding=None,
        data_format=None,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


@utils.register_keras_custom_object
class QuantLocallyConnected1D(QuantizerBase, tf.keras.layers.LocallyConnected1D):
    """Locally-connected quantized layer for 1D inputs.

    The `QuantLocallyConnected1D` layer works similarly to the `QuantConv1D` layer,
    except that weights are unshared, that is, a different set of filters is applied
    at each different patch of the input. `input_quantizer` and `kernel_quantizer`
    are the element-wise quantization functions to use. If both quantization functions
    are `None` this layer is equivalent to `LocallyConnected1D`.

    !!! example
        ```python
        # apply a unshared weight convolution 1d of length 3 to a sequence with
        # 10 timesteps, with 64 output filters
        model = Sequential()
        model.add(QuantLocallyConnected1D(64, 3, input_shape=(10, 32)))
        # now model.output_shape == (None, 8, 64)
        # add a new conv1d on top
        model.add(QuantLocallyConnected1D(32, 3))
        # now model.output_shape == (None, 6, 32)
        ```

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the stride
            length of the convolution. Specifying any stride value != 1 is incompatible
            with specifying any `dilation_rate` value != 1.
        padding: Currently only supports `"valid"` (case-insensitive).
            `"same"` may be supported in the future.
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs. `channels_last` corresponds
            to inputs with shape `(batch, length, channels)` while `channels_first`
            corresponds to inputs with shape `(batch, channels, length)`. It defaults
            to the `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be "channels_last".
        activation: Activation function to use. If you don't specify anything,
            no activation is applied (`a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        input_quantizer: Quantization function applied to the input of the layer.
        kernel_quantizer: Quantization function applied to the `kernel` weights matrix.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.
        implementation: implementation mode, either `1` or `2`.
            `1` loops over input spatial locations to perform the forward pass.
            It is memory-efficient but performs a lot of (small) ops.

            `2` stores layer weights in a dense but sparsely-populated 2D matrix
            and implements the forward pass as a single matrix-multiply. It uses
            a lot of RAM but performs few (large) ops.

            Depending on the inputs, layer parameters, hardware, and
            `tf.executing_eagerly()` one implementation can be dramatically faster
            (e.g. 50X) than another.

            It is recommended to benchmark both in the setting of interest to pick
            the most efficient one (in terms of speed and memory usage).

            Following scenarios could benefit from setting `implementation=2`:

            - eager execution;
            - inference;
            - running on CPU;
            - large amount of RAM available;
            - small models (few filters, small kernel);
            - using `padding=same` (only possible with `implementation=2`).

    # Input shape
        3D tensor with shape: `(batch_size, steps, input_dim)`

    # Output shape
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        implementation=1,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            implementation=implementation,
            **kwargs,
        )


@utils.register_keras_custom_object
class QuantLocallyConnected2D(QuantizerBase, tf.keras.layers.LocallyConnected2D):
    """Locally-connected quantized layer for 2D inputs.

    The `QuantLocallyConnected2D` layer works similarly to the `QuantConv2D` layer,
    except that weights are unshared, that is, a different set of filters is applied
    at each different patch of the input. `input_quantizer` and `kernel_quantizer`
    are the element-wise quantization functions to use. If both quantization functions
    are `None` this layer is equivalent to `LocallyConnected2D`.

    !!! example
        ```python
        # apply a 3x3 unshared weights convolution with 64 output filters on a
        32x32 image
        # with `data_format="channels_last"`:
        model = Sequential()
        model.add(QuantLocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
        # now model.output_shape == (None, 30, 30, 64)
        # notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64
        parameters

        # add a 3x3 unshared weights convolution on top, with 32 output filters:
        model.add(QuantLocallyConnected2D(32, (3, 3)))
        # now model.output_shape == (None, 28, 28, 32)
        ```

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window. Can be a single integer to
            specify the same value for all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides of the
            convolution along the width and height. Can be a single integer to specify
            the same value for all spatial dimensions.
        padding: Currently only support `"valid"` (case-insensitive).
            `"same"` will be supported in future.
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs. `channels_last` corresponds to
            inputs with shape `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape `(batch, channels, height, width)`. It
            defaults to the `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be "channels_last".
        activation: Activation function to use. If you don't specify anything,
            no activation is applied (`a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        input_quantizer: Quantization function applied to the input of the layer.
        kernel_quantizer: Quantization function applied to the `kernel` weights matrix.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.
        implementation: implementation mode, either `1` or `2`.
            `1` loops over input spatial locations to perform the forward pass.
            It is memory-efficient but performs a lot of (small) ops.

            `2` stores layer weights in a dense but sparsely-populated 2D matrix
            and implements the forward pass as a single matrix-multiply. It uses
            a lot of RAM but performs few (large) ops.

            Depending on the inputs, layer parameters, hardware, and
            `tf.executing_eagerly()` one implementation can be dramatically faster
            (e.g. 50X) than another.

            It is recommended to benchmark both in the setting of interest to pick
            the most efficient one (in terms of speed and memory usage).

            Following scenarios could benefit from setting `implementation=2`:

            - eager execution;
            - inference;
            - running on CPU;
            - large amount of RAM available;
            - small models (few filters, small kernel);
            - using `padding=same` (only possible with `implementation=2`).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        implementation=1,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            implementation=implementation,
            **kwargs,
        )
