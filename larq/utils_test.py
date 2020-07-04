from larq import utils


def test_memory_as_readable_str():
    correct_strings = [  # 2^i bits, from i = 0 to 74
        "0.12 B",
        "0.25 B",
        "0.50 B",
        "1.00 B",
        "2.00 B",
        "4.00 B",
        "8.00 B",
        "16.00 B",
        "32.00 B",
        "64.00 B",
        "128.00 B",
        "256.00 B",
        "512.00 B",
        "1.00 KiB",
        "2.00 KiB",
        "4.00 KiB",
        "8.00 KiB",
        "16.00 KiB",
        "32.00 KiB",
        "64.00 KiB",
        "128.00 KiB",
        "256.00 KiB",
        "512.00 KiB",
        "1.00 MiB",
        "2.00 MiB",
        "4.00 MiB",
        "8.00 MiB",
        "16.00 MiB",
        "32.00 MiB",
        "64.00 MiB",
        "128.00 MiB",
        "256.00 MiB",
        "512.00 MiB",
        "1.00 GiB",
        "2.00 GiB",
        "4.00 GiB",
        "8.00 GiB",
        "16.00 GiB",
        "32.00 GiB",
        "64.00 GiB",
        "128.00 GiB",
        "256.00 GiB",
        "512.00 GiB",
        "1,024.00 GiB",
    ]

    for i, correct_string in enumerate(correct_strings):
        assert utils.memory_as_readable_str(2 ** i) == correct_string


def test_set_precision():
    @utils.set_precision(8)
    def toy_quantizer(x):
        return x

    assert toy_quantizer.precision == 8
