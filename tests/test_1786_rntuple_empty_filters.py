# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import pytest

import uproot


@pytest.fixture
def ntuple(tmp_path):
    path = tmp_path / "empty-filter.root"
    with uproot.recreate(path) as output:
        output.mkrntuple(
            "ntuple",
            {"top": [1, 2], "struct": [{"field": 1}, {"field": 2}]},
        )
    with uproot.open(path) as source:
        yield source["ntuple"]


@pytest.mark.parametrize("field_path", [None, "struct"])
@pytest.mark.parametrize("method", ["num_entries_for", "arrays", "iterate"])
def test_empty_filter_raises_for_all_read_methods(ntuple, field_path, method):
    target = ntuple if field_path is None else ntuple[field_path]

    with pytest.raises(
        ValueError,
        match="no fields matched the given filters: filter_name='does_not_exist'",
    ):
        if method == "num_entries_for":
            target.num_entries_for("1 kB", filter_name="does_not_exist")
        elif method == "arrays":
            target.arrays(filter_name="does_not_exist")
        else:
            list(target.iterate(filter_name="does_not_exist", step_size=1))
