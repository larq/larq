from sys import stdout
import numpy as np


def _terminal_supports_unicode():
    return hasattr(stdout, "encoding") and stdout.encoding in ("utf-8", "UTF-8", "UTF8")


def _get_delimiter(type_="thin"):
    if _terminal_supports_unicode():
        return "━" if type_ == "thick" else "─"
    return "=" if type_ == "thick" else "-"


def _count_params(weights):
    """Count the total number of scalars composing the weights.

    # Arguments
    weights: An iterable containing the weights on which to compute params

    # Returns
    The total number of scalars composing the weights
    """
    return int(sum(np.prod(w.shape.as_list()) for w in weights))


def _get_output_shape(layer):
    try:
        return tuple(dim if dim else -1 for dim in layer.output_shape)
    except AttributeError:
        return "multiple"
    except RuntimeError:  # output_shape unknown in Eager mode.
        return "?"


def _count_binarized_weights(layer):
    if hasattr(layer, "quantized_latent_weights"):
        return _count_params(layer.quantized_latent_weights)
    return 0


def _count_fp_weights(layer):
    if hasattr(layer, "quantized_latent_weights"):
        return int(
            sum(
                np.prod(w.shape.as_list())
                for w in layer.weights
                if w not in layer.quantized_latent_weights
            )
        )
    return layer.count_params()


def _bit_to_kB(bit_value):
    return bit_value / 8 / 1024


def _memory_weights(layer):
    num_fp_params = _count_fp_weights(layer)
    num_binarized_params = _count_binarized_weights(layer)
    fp32 = 32  # Multiply float32 params by 32 to get bit value
    total_layer_mem_in_bits = (num_fp_params * fp32) + (num_binarized_params)
    return _bit_to_kB(total_layer_mem_in_bits)


def summary(model, line_length=None, positions=None, print_fn=None):
    """Prints a string summary of the network.

    # Arguments
    model: `tf.keras` model instance.
    line_length: Total length of printed lines
        (e.g. set this to adapt the display to different terminal window sizes).
    positions: Relative or absolute positions of log elements in each line.
        If not provided, defaults to `[0.38, 0.62, 0.74, 0.88, 1.0]`.
    print_fn: Print function to use. Defaults to `print`. You can set it to a custom
        function in order to capture the string summary.

    # Raises
    ValueError: if called before the model is built.
    """

    if not model.built:
        raise ValueError(
            "This model has not yet been built. Build the model first by calling "
            "`model.build()` or calling `model.fit()` with some data, or specify an "
            "`input_shape` argument in the first layer(s) for automatic build."
        )

    header = ("Layer", "Outputs", "# 1-bit", "# 32-bit", "Mem (kB)")

    model._check_trainable_weights_consistency()
    if hasattr(model, "_collected_trainable_weights"):
        trainable_count = _count_params(model._collected_trainable_weights)
    else:
        trainable_count = _count_params(model.trainable_weights)
    non_trainable_count = _count_params(model.non_trainable_weights)

    if print_fn is None:
        print_fn = print

    line_length = line_length or 88
    positions = positions or [0.38, 0.62, 0.74, 0.88, 1.0]
    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]

    def print_row(fields, positions):
        line = ""
        for i, (field, position) in enumerate(zip(fields, positions)):
            field = f"{field:.2f}" if type(field) == float else str(field)
            if i == 0:
                line += field
                line += " " * (position - len(line))
                line = line[: position - 1] + " "
            else:
                line += " " * (position - len(line) - len(field)) + field
                line = line[:position]

        print_fn(line)

    print_fn(_get_delimiter("thick") * line_length)
    print_row(header, positions)
    print_fn(_get_delimiter() * line_length)

    amount_binarized = amount_full_precision = total_memory = 0
    for layer in model.layers:
        n_bin = _count_binarized_weights(layer)
        n_fp = _count_fp_weights(layer)
        memory = _memory_weights(layer)
        amount_binarized += n_bin
        amount_full_precision += n_fp
        total_memory += memory
        print_row(
            (layer.name, _get_output_shape(layer), n_bin, n_fp, memory), positions
        )
    print_fn(_get_delimiter() * line_length)
    print_row(
        ("Total", "", amount_binarized, amount_full_precision, total_memory), positions
    )
    print_fn(_get_delimiter("thick") * line_length)
    print_fn(f"Total params: {trainable_count + non_trainable_count}")
    print_fn(f"Trainable params: {trainable_count}")
    print_fn(f"Non-trainable params: {non_trainable_count}")

    float32_equiv = _bit_to_kB((amount_binarized + amount_full_precision) * 32)
    compression_ratio = float32_equiv / total_memory

    print_fn(_get_delimiter() * line_length)
    print_fn(f"Float-32 Equivalent: {float32_equiv / 1024:.2f} MB")
    print_fn(f"Compression of Memory: {compression_ratio:.2f}")
    print_fn(_get_delimiter("thick") * line_length)
