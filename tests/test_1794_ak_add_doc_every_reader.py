# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import pytest
import skhep_testdata

import uproot

from tests.test_1785_rntuple_ak_add_doc_every_reader import (  # noqa: F401 (fixture)
    _form,
    _lazy,
    documented_rntuple,
)

TTREE = skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
COFFEA_DOC = {"__doc__": "title", "typename": "typename"}


@pytest.mark.parametrize("reader", ["eager", "dask-delayed", "dask-open", "graphed"])
def test_title_reads_the_rntuple_description(documented_rntuple, reader):
    ((path, name),) = documented_rntuple.items()
    ntuple = uproot.open(path)[name]
    described = ntuple.arrays(ak_add_doc={"__doc__": "description"}).layout.form
    if reader == "eager":
        form = ntuple.arrays(ak_add_doc={"__doc__": "title"}).layout.form
    else:
        form = _form(
            reader, _lazy(reader, documented_rntuple, ak_add_doc={"__doc__": "title"})
        )
    assert form.content("x").parameters["__doc__"] == "an int"
    assert form.is_equal_to(described, all_parameters=True)


def test_an_open_ttree_takes_a_dict_doc_in_dask():
    pytest.importorskip("dask_awkward")
    tree = uproot.open(TTREE)["tree"]
    eager = tree.arrays(["I32"], ak_add_doc={"__doc__": "title"}).layout.form
    lazy = uproot.dask(
        tree, filter_name=["I32"], open_files=False, ak_add_doc={"__doc__": "title"}
    )
    assert lazy.layout.form.is_equal_to(eager, all_parameters=True)


def test_rntuple_and_its_fields_have_a_title(documented_rntuple):
    ((path, name),) = documented_rntuple.items()
    ntuple = uproot.open(path)[name]
    assert ntuple.title == "top"
    assert ntuple["x"].title == "an int"
    assert ntuple["r.a"].title == "a subfield"


def test_virtual_string_branch_keeps_its_type_with_docs():
    form = (
        uproot.open(TTREE)["tree"]
        .arrays(filter_name=["StdStr"], ak_add_doc=True, virtual=True)
        .layout.form
    )
    assert form.content("StdStr").parameters == {
        "__array__": "string",
        "__doc__": "StdStr",
    }


def test_virtual_top_level_doc_reads_the_dict_attribute():
    form = (
        uproot.open(TTREE)["tree"]
        .arrays(filter_name=["I32"], ak_add_doc={"__doc__": "name"}, virtual=True)
        .layout.form
    )
    assert form.parameters == {"__doc__": "tree"}


def test_eager_ttree_takes_a_coffea_dict():
    layout = uproot.open(TTREE)["tree"].arrays(["I32"], ak_add_doc=COFFEA_DOC).layout
    assert layout.parameters == {"__doc__": "my tree title"}
    assert layout.content("I32").parameters == {"__doc__": "I32", "typename": "int32_t"}


def test_true_and_a_dict_agree_on_empty_descriptions(documented_rntuple):
    ((path, name),) = documented_rntuple.items()
    ntuple = uproot.open(path)[name]
    titled = ntuple.arrays(ak_add_doc={"__doc__": "title"}).layout.form
    form = ntuple.arrays(ak_add_doc=True).layout.form
    assert form.content("s").parameters == {"__array__": "string", "__doc__": ""}
    assert form.content("r").content("b").parameters == {"__doc__": ""}
    assert form.is_equal_to(titled, all_parameters=True)


def test_untitled_ttree_keeps_its_empty_doc(tmp_path):
    path = str(tmp_path / "untitled.root")
    with uproot.recreate(path) as file:
        file.mktree("t", {"a": "int64"})
        file["t"].extend({"a": [1, 2]})
    layout = uproot.open(path)["t"].arrays(ak_add_doc=True).layout
    assert layout.parameters == {"__doc__": ""}
    assert layout.content("a").parameters == {"__doc__": "a/L"}


@pytest.fixture(params=["ttree", "rntuple"])
def source(request, documented_rntuple):
    if request.param == "ttree":
        return {TTREE: "tree"}, {"filter_name": ["I32", "StdStr"]}, "my tree title"
    return documented_rntuple, {}, "top"


@pytest.mark.parametrize("ak_add_doc", [True, COFFEA_DOC])
@pytest.mark.parametrize("reader", ["virtual", "dask-delayed", "dask-open", "graphed"])
def test_every_reader_builds_the_eager_form(source, reader, ak_add_doc):
    files, options, title = source
    ((path, name),) = files.items()
    tree = uproot.open(path)[name]
    eager = tree.arrays(**options, ak_add_doc=ak_add_doc).layout.form
    if reader == "virtual":
        form = tree.arrays(**options, ak_add_doc=ak_add_doc, virtual=True).layout.form
    else:
        form = _form(reader, _lazy(reader, files, ak_add_doc=ak_add_doc, **options))
    assert eager.parameters == {"__doc__": title}
    assert form.is_equal_to(eager, all_parameters=True)
