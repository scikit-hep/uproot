# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import skhep_testdata

import uproot
from uproot.behaviors.RNTuple import iterate as rn_iterate


def test_rfield_getitem_nonexistent_raises_key_in_file_error():
    """field['.nonexistent'] must raise KeyInFileError, not AttributeError."""
    filename = skhep_testdata.data_path("test_nested_structs_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        ntuple = f["ntuple"]
        field = next(iter(ntuple.values()))
        with pytest.raises(uproot.KeyInFileError, match="not found: '.nonexistent'"):
            field[".nonexistent"]


def test_rntuple_iterate_multifile_global_offsets():
    """report.global_entry_start/stop must be continuous across files."""
    filename = skhep_testdata.data_path("test_int_float_rntuple_v1-0-0-0.root")
    results = list(
        rn_iterate([f"{filename}:ntuple", f"{filename}:ntuple"], report=True)
    )
    assert len(results) == 2, "expected one batch per file"
    _, rep0 = results[0]
    _, rep1 = results[1]
    # Second file's global start must equal first file's global stop
    assert rep0.global_entry_start == 0
    assert rep0.global_entry_stop == 10
    assert rep1.global_entry_start == 10
    assert rep1.global_entry_stop == 20
    # Per-file (tree) entries restart at 0 for each file
    assert rep0.tree_entry_start == 0
    assert rep1.tree_entry_start == 0
