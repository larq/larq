import numpy as np
import tensorflow as tf

import larq as lq


def _eval_tensor(tensor):
    if tensor is None:
        return None
    elif callable(tensor):
        return _eval_helper(tensor())
    else:
        return tensor.numpy()


def _eval_helper(tensors):
    if tensors is None:
        return None
    return tf.nest.map_structure(_eval_tensor, tensors)


def evaluate(tensors):
    if tf.executing_eagerly():
        return _eval_helper(tensors)
    else:
        sess = tf.compat.v1.get_default_session()
        return sess.run(tensors)


def generate_real_values_with_zeros(low=-2, high=2, shape=(4, 10)):
    real_values = np.random.uniform(low, high, shape)
    real_values = np.insert(real_values, 1, 0, axis=1)
    return real_values


def get_small_bnn_model(input_dim, num_hidden, output_dim, trainable_bn=True):
    model = tf.keras.models.Sequential()
    model.add(
        lq.layers.QuantDense(
            units=num_hidden,
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            activation="relu",
            input_shape=(input_dim,),
            use_bias=False,
        )
    )
    model.add(tf.keras.layers.BatchNormalization(trainable=trainable_bn))
    model.add(
        lq.layers.QuantDense(
            units=output_dim,
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            input_quantizer="ste_sign",
            activation="softmax",
            use_bias=False,
        )
    )
    return model


def random_input(shape):
    for i, dim in enumerate(shape):
        if dim is None:
            shape[i] = np.random.randint(1, 4)
    data = 10 * np.random.random(shape) - 0.5
    return data.astype("float32")


# This is a fork of https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/testing_utils.py#L72
# as recommended in https://github.com/tensorflow/tensorflow/issues/28601#issuecomment-492810252
def layer_test(
    layer_cls,
    kwargs=None,
    input_shape=None,
    input_dtype=None,
    input_data=None,
    expected_output=None,
    expected_output_dtype=None,
    should_run_eagerly=False,
):
    """Test routine for a layer with a single input and single output.
    Arguments:
      layer_cls: Layer class object.
      kwargs: Optional dictionary of keyword arguments for instantiating the
        layer.
      input_shape: Input shape tuple.
      input_dtype: Data type of the input data.
      input_data: Numpy array of input data.
      expected_output: Shape tuple for the expected shape of the output.
      expected_output_dtype: Data type expected for the output.
    Returns:
      The output data (Numpy array) returned by the layer, for additional
      checks to be done by the calling code.
    Raises:
      ValueError: if `input_shape is None`.
    """
    if input_data is None:
        if input_shape is None:
            raise ValueError("input_shape is None")
        if not input_dtype:
            input_dtype = "float32"
        input_data_shape = list(input_shape)
        for i, e in enumerate(input_data_shape):
            if e is None:
                input_data_shape[i] = np.random.randint(1, 4)
        input_data = 10 * np.random.random(input_data_shape)
        if input_dtype[:5] == "float":
            input_data -= 0.5
        input_data = input_data.astype(input_dtype)
    elif input_shape is None:
        input_shape = input_data.shape
    if input_dtype is None:
        input_dtype = input_data.dtype
    if expected_output_dtype is None:
        expected_output_dtype = input_dtype

    # instantiation
    kwargs = kwargs or {}
    layer = layer_cls(**kwargs)

    # test get_weights , set_weights at layer level
    weights = layer.get_weights()
    layer.set_weights(weights)

    # test in functional API
    x = tf.keras.layers.Input(shape=input_shape[1:], dtype=input_dtype)
    y = layer(x)
    if tf.keras.backend.dtype(y) != expected_output_dtype:
        raise AssertionError(
            "When testing layer %s, for input %s, found output "
            "dtype=%s but expected to find %s.\nFull kwargs: %s"
            % (
                layer_cls.__name__,
                x,
                tf.keras.backend.dtype(y),
                expected_output_dtype,
                kwargs,
            )
        )
    # check shape inference
    model = tf.keras.models.Model(x, y)
    expected_output_shape = tuple(
        layer.compute_output_shape(tf.TensorShape(input_shape)).as_list()
    )
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    for expected_dim, actual_dim in zip(expected_output_shape, actual_output_shape):
        if expected_dim is not None:
            if expected_dim != actual_dim:
                raise AssertionError(
                    "When testing layer %s, for input %s, found output_shape="
                    "%s but expected to find %s.\nFull kwargs: %s"
                    % (
                        layer_cls.__name__,
                        x,
                        actual_output_shape,
                        expected_output_shape,
                        kwargs,
                    )
                )
    if expected_output is not None:
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = tf.keras.models.Model.from_config(model_config)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        output = recovered_model.predict(input_data)
        np.testing.assert_allclose(output, actual_output, rtol=2e-3)

    # Recreate layer to prevent layer metrics from being configured multiple times.
    layer = layer_cls(**kwargs)
    # test training mode (e.g. useful for dropout tests)
    # Rebuild the model to avoid the graph being reused between predict() and
    # train(). This was causing some error for layer with Defun as it body.
    # See b/120160788 for more details. This should be mitigated after 2.0.
    model = tf.keras.models.Model(x, layer(x))
    model.compile(
        "rmsprop",
        "mse",
        weighted_metrics=["acc"],
        run_eagerly=should_run_eagerly,
    )
    model.train_on_batch(input_data, actual_output)

    # test as first layer in Sequential API
    layer_config = layer.get_config()
    layer_config["batch_input_shape"] = input_shape
    layer = layer.__class__.from_config(layer_config)

    model = tf.keras.models.Sequential()
    model.add(layer)
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    for expected_dim, actual_dim in zip(expected_output_shape, actual_output_shape):
        if expected_dim is not None:
            if expected_dim != actual_dim:
                raise AssertionError(
                    "When testing layer %s **after deserialization**, "
                    "for input %s, found output_shape="
                    "%s but expected to find inferred shape %s.\nFull kwargs: %s"
                    % (
                        layer_cls.__name__,
                        x,
                        actual_output_shape,
                        expected_output_shape,
                        kwargs,
                    )
                )
    if expected_output is not None:
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = tf.keras.models.Sequential.from_config(model_config)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        output = recovered_model.predict(input_data)
        np.testing.assert_allclose(output, actual_output, rtol=2e-3)

    # for further checks in the caller function
    return actual_output
