import numpy as np
from tabulate import tabulate


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


def _bit_to_MB(bit_value):
    bittobyte = 1.0 / 8.0
    mbconverter = 1024 ** 2
    return bit_value * bittobyte / mbconverter
bit_to_byte_ratio = 1.0 / 8.0
byte_to_mega_bytes_ratio = 1.0 / (1024**2)
return bit_value * bit_to_byte_ratio * byte_to_mega_bytes_ratio

def _memory_weights(layer):
    fp_params = _count_fp_weights(layer)
    b_params = _count_binarized_weights(layer)
    fp32 = 32  # Multiply float32 params by 32 to get bit value
    memory_bits = (fp_params * fp32) + (b_params)
    return _bit_to_MB(memory_bits)


def summary(model, tablefmt="simple", print_fn=None):
    """Prints a string summary of the network.

    # Arguments
    model: `tf.keras` model instance.
    tablefmt: Supported table formats are: `fancy_grid`, `github`, `grid`, `html`,
        `jira`, `latex`, `latex_booktabs`, `latex_raw`, `mediawiki`, `moinmoin`,
        `orgtbl`, `pipe`, `plain`, `presto`, `psql`, `rst`, `simple`, `textile`,
        `tsv`, `youtrac`.
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

    header = ("Layer", "Outputs", "# 1-bit", "# 32-bit", "Memory (MB)")
    table = [
        [
            layer.name,
            _get_output_shape(layer),
            _count_binarized_weights(layer),
            _count_fp_weights(layer),
            _memory_weights(layer),
        ]
        for layer in model.layers
    ]

    amount_binarized = sum(r[2] for r in table)
    amount_full_precision = sum(r[3] for r in table)
    total_memory = sum(r[4] for r in table)

    table.append(["Total", None, amount_binarized, amount_full_precision, total_memory])

    model._check_trainable_weights_consistency()
    if hasattr(model, "_collected_trainable_weights"):
        trainable_count = _count_params(model._collected_trainable_weights)
    else:
        trainable_count = _count_params(model.trainable_weights)
    non_trainable_count = _count_params(model.non_trainable_weights)

    if print_fn is None:
        print_fn = print

    print_fn(tabulate(table, headers=header, tablefmt=tablefmt))
    print_fn()
    print_fn(f"Total params: {trainable_count + non_trainable_count}")
    print_fn(f"Trainable params: {trainable_count}")
    print_fn(f"Non-trainable params: {non_trainable_count}")

    float32_equiv = _bit_to_MB((amount_binarized + amount_full_precision) * 32)
    compression_ratio = float32_equiv / total_memory

    print_fn(f"Float-32 Equivalent: {float32_equiv:.2f} MB")
    print_fn(f"Compression of Memory: {compression_ratio:.2f}")
    print_fn()
