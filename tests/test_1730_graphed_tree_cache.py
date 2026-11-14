# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
"""A worker reads a file's ``TTree`` once, not once per partition."""

from __future__ import annotations

import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")

from graphed.core.execution import LocalResources  # noqa: E402


def test_partitions_of_one_file_share_the_ttree():
    array = uproot.graphed({skhep_testdata.data_path("uproot-Zmumu.root"): "events"})
    _nid, source = uproot._graphed._uproot_source(array, "test")
    resources = LocalResources()
    first, second = (
        source.open_tree(partition, resources)
        for partition in source.partitions(steps_per_file=2)
    )
    assert resources.open_count == 1
    assert second is first
