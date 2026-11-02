# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
import pytest
import skhep_testdata

import uproot

readers = pytest.mark.parametrize("reader", ["dask-delayed", "dask-open", "graphed"])
add_docs = pytest.mark.parametrize(
    "ak_add_doc", [True, {"__doc__": "description", "typename": "typename"}]
)


def _lazy(reader, files, **options):
    if reader == "graphed":
        pytest.importorskip("graphed")
        return uproot.graphed(files, **options)
    pytest.importorskip("dask_awkward")
    return uproot.dask(files, open_files=reader == "dask-open", **options)


def _form(reader, array):
    if reader == "graphed":
        return array.session.form(array).tt.layout.form
    return array.layout.form


@pytest.fixture(scope="module")
def documented_rntuple(tmp_path_factory):
    """Documented top record, field and subfield, plus an undocumented string field."""
    path = str(tmp_path_factory.mktemp("docs") / "docs.root")
    numbers = ak.to_layout([1, 2])
    struct = ak.contents.RecordArray(
        [numbers.copy(parameters={"__doc__": "a subfield"}), numbers],
        ["a", "b"],
        parameters={"__doc__": "a struct"},
    )
    data = ak.contents.RecordArray(
        [
            numbers.copy(parameters={"__doc__": "an int"}),
            ak.to_layout(["u", "v"]),
            struct,
        ],
        ["x", "s", "r"],
        parameters={"__doc__": "top"},
    )
    with uproot.recreate(path) as file:
        file.mkrntuple("ntpl", ak.Array(data))
    return {path: "ntpl"}


@add_docs
@readers
def test_rntuple_form_matches_eager(documented_rntuple, reader, ak_add_doc):
    ((path, name),) = documented_rntuple.items()
    eager = uproot.open(path)[name].arrays(ak_add_doc=ak_add_doc)
    array = _lazy(reader, documented_rntuple, ak_add_doc=ak_add_doc)
    assert _form(reader, array).is_equal_to(eager.layout.form, all_parameters=True)
    if reader != "graphed":
        computed = array.compute()
        assert computed.layout.form.is_equal_to(eager.layout.form, all_parameters=True)
        assert ak.array_equal(computed, eager)


@readers
def test_rntuple_missing_doc_attribute_fails_at_build(documented_rntuple, reader):
    with pytest.raises(AttributeError, match="no_such_attribute"):
        _lazy(reader, documented_rntuple, ak_add_doc={"__doc__": "no_such_attribute"})


@readers
def test_rntuple_base_form_has_no_typenames(documented_rntuple, reader):
    seen = []

    def form_mapping(base_form):
        seen.append(base_form)
        return uproot._dask._mapping_of(None, base_form)

    _lazy(reader, documented_rntuple, form_mapping=form_mapping)
    assert seen
    assert "typenames" not in seen[0].parameters


@pytest.mark.parametrize(
    "ak_add_doc", [True, {"__doc__": "title"}, {"__doc__": "name"}]
)
@readers
def test_ttree_string_branch_keeps_its_type_with_docs(reader, ak_add_doc):
    path = skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    keys = ["StdStr", "I32"]
    eager = uproot.open(path)["tree"].arrays(keys, ak_add_doc=ak_add_doc).layout.form
    form = _form(
        reader, _lazy(reader, {path: "tree"}, filter_name=keys, ak_add_doc=ak_add_doc)
    )
    assert form.content("StdStr").parameters["__array__"] == "string"
    assert form.is_equal_to(eager, all_parameters=True)
