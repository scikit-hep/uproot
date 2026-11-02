# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata
import numpy as np

import uproot


def truncate_float(value, bits):
    a = np.float32(value).view(np.uint32)
    a &= np.uint32(0xFFFFFFFF) << (32 - bits)
    return a.astype(np.uint32).view(np.float32)


def quantize_float(value, bits, min, max):
    min = np.float32(min)
    max = np.float32(max)
    if value < min or value > max:
        raise ValueError(f"Value {value} is out of range [{min}, {max}]")
    scaled_value = (value - min) * (2**bits - 1) / (max - min)
    int_value = np.round(scaled_value)
    quantized_float = min + int_value * (max - min) / ((1 << bits) - 1)
    return quantized_float.astype(np.float32)


@pytest.mark.parametrize(
    ("entry", "true_truncated", "true_quantized"),
    [
        (0, 1.23456789, 1.23456789),
        (1, 1.4660155e13, 1.6666666),
        (2, -6.2875986e-22, -6.2875986e-22),
        (3, -1.9060668, -1.9060668),
    ],
)
def test_custom_floats(entry, true_truncated, true_quantized):
    filename = skhep_testdata.data_path("test_float_types_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        row = f["ntuple"].arrays()[entry]

    min_value = -2.0
    max_value = 3.0

    for bits in (10, 16, 24, 31):
        assert row[f"trunc{bits}"] == truncate_float(true_truncated, bits)
    for bits in (1, 8, 16, 20, 24, 25, 32):
        expected = quantize_float(true_quantized, bits, min_value, max_value)
        # one float32 step at this range, much less than the quantization of 20 bits
        assert row[f"quant{bits}"] == pytest.approx(expected, rel=0, abs=5e-7)


def test_multiple_representations():
    filename = skhep_testdata.data_path(
        "test_multiple_representations_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        assert len(obj.page_link_list) == 3
        # The zeroth representation is active in clusters 0 and 2, but not in cluster 1
        assert not obj.page_link_list[0][0].suppressed
        assert obj.page_link_list[1][0].suppressed
        assert not obj.page_link_list[2][0].suppressed

        arrays = obj.arrays()

        assert arrays.real.tolist() == [1, 2, 3]
