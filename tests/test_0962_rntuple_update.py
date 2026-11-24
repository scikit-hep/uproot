# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import uproot
import awkward as ak
import skhep_testdata
import numpy as np


def test_new_support_RNTuple_split_int32_reading():
    with uproot.open(
        skhep_testdata.data_path("test_int_5e4_rntuple_v1-0-0-0.root")
    ) as f:
        obj = f["ntuple"]
        df = obj.arrays()
        assert len(df) == 5e4
        assert len(df.one_integers) == 5e4
        assert np.all(df.one_integers == np.arange(5e4 + 1)[::-1][:-1])


@pytest.mark.parametrize(
    ("column", "bits"),
    [("int16", 16), ("int32", 32), ("int64", 64)],
)
def test_new_support_RNTuple_split_int_extended_range(column, bits):
    half = 2 ** (bits - 2)
    values = [0, 1, -1, half, -half, 2 * half - 1, -2 * half]
    with uproot.open(
        skhep_testdata.data_path("test_splitint_rntuple_v1-0-1-0.root")
    ) as f:
        result = f["ntuple"][column].array().tolist()
    assert result == values


def test_new_support_RNTuple_bit_bool_reading():
    with uproot.open(skhep_testdata.data_path("test_bit_rntuple_v1-0-0-0.root")) as f:
        obj = f["ntuple"]
        df = obj.arrays()
        assert df.one_bit.tolist() == [i % 3 == 0 for i in range(10)]


def test_new_support_RNTuple_split_int16_reading():
    with uproot.open(
        skhep_testdata.data_path("test_int_multicluster_rntuple_v1-0-0-0.root")
    ) as f:
        obj = f["ntuple"]
        assert obj.num_entries == 10**8

        field_metadata = obj.get_field_metadata(0)
        assert field_metadata.dtype == np.dtype("int16")
        assert field_metadata.split
        assert field_metadata.zigzag

        # Reading all of the 191 pages takes a minute, but they are stored only once
        # each: the first half of the entries are 2 and the rest are 1, so that page 95
        # has both
        pages = obj.page_link_list[0][0].pages
        assert sum(page.num_elements for page in pages) == 10**8
        assert {page.locator.offset for page in pages} == {
            pages[i].locator.offset for i in (0, 95, 96, 190)
        }
        change = 5 * 10**7 - 95 * pages[0].num_elements

        def read_page(page_idx):
            out = np.empty(pages[page_idx].num_elements, np.dtype("int16"))
            obj.read_page(out, 0, 0, page_idx, field_metadata)
            obj.post_process(out, field_metadata)
            return out

        assert np.all(read_page(0) == 2)
        middle = read_page(95)
        assert np.all(middle[:change] == 2)
        assert np.all(middle[change:] == 1)
        assert np.all(read_page(96) == 1)
        assert np.all(read_page(190) == 1)
