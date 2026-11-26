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


@pytest.mark.parametrize("method", ["num_entries_for", "arrays", "iterate"])
@pytest.mark.parametrize(
    "filter_kwargs", [{"filter_name": "*"}, {"filter_typename": "*"}]
)
def test_matching_filter_on_a_leaf_field_still_reads(ntuple, method, filter_kwargs):
    # A leaf field has no subfields, so counting keys() would reject even a filter that
    # matches the field itself. The selected form always includes the field.
    leaf = ntuple["top"]

    if method == "num_entries_for":
        assert leaf.num_entries_for("1 kB", **filter_kwargs) > 0
    elif method == "arrays":
        assert leaf.arrays(**filter_kwargs).tolist() == [{"top": 1}, {"top": 2}]
    else:
        chunks = [
            chunk.tolist() for chunk in leaf.iterate(step_size=1, **filter_kwargs)
        ]
        assert chunks == [[{"top": 1}], [{"top": 2}]]


def test_expressions_on_a_leaf_field_still_reads(ntuple):
    # expressions is copied into filter_name, so it hit the same rejection.
    assert ntuple["top"].arrays(expressions="top").tolist() == [{"top": 1}, {"top": 2}]


def test_message_lists_only_the_filters_that_were_given(ntuple):
    with pytest.raises(ValueError) as excinfo:
        ntuple.arrays(filter_name="does_not_exist")

    message = str(excinfo.value)
    assert message == (
        "no fields matched the given filters: filter_name='does_not_exist'"
    )
    assert "no_filter" not in message
    assert "unset" not in message


def test_num_entries_for_on_an_empty_ntuple_is_unchanged(tmp_path):
    # Without the empty-form guard this returned 1 instead of None.
    path = tmp_path / "no-fields.root"
    with uproot.recreate(path) as output:
        output.mkrntuple("ntuple", {})

    with uproot.open(path) as source:
        assert source["ntuple"].num_entries_for("1 kB") is None
