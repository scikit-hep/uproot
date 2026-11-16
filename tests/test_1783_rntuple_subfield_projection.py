# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import os

import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")


def _rntuple(tmp_path, arr):
    path = os.path.join(tmp_path, "nested.root")
    with uproot.recreate(path) as f:
        f["nt"] = arr
    return path + ":nt", arr


def _nested_rntuple(tmp_path):
    return _rntuple(
        tmp_path,
        ak.Array(
            [
                {
                    "jets": [{"pt": float(j), "eta": 0.1 * j} for j in range(k % 3)],
                    "rec": {"x": k, "y": 2 * k, "lst": [k] * (k % 2)},
                    "z": 3 * k,
                }
                for k in range(6)
            ]
        ),
    )


def _jets_with_hits(tmp_path):
    return _rntuple(
        tmp_path,
        ak.Array(
            [
                {
                    "jets": [
                        {"hits": [0.5] * j, "pt": float(j), "eta": 0.1 * j}
                        for j in range(k % 3)
                    ]
                }
                for k in range(6)
            ]
        ),
    )


def _columns_read(monkeypatch):
    from uproot.models.RNTuple import Model_ROOT_3a3a_RNTuple

    read = set()
    real = Model_ROOT_3a3a_RNTuple.read_cluster_pages

    def logged(self, cluster_idx, col_idx, *args, **kwargs):
        read.add(col_idx)
        return real(self, cluster_idx, col_idx, *args, **kwargs)

    monkeypatch.setattr(Model_ROOT_3a3a_RNTuple, "read_cluster_pages", logged)
    return read


@pytest.mark.parametrize("open_files", [True, False])
def test_dask_reads_only_the_subfields_it_needs(tmp_path, monkeypatch, open_files):
    dak = pytest.importorskip("dask_awkward")
    where, arr = _nested_rntuple(tmp_path)
    ev = uproot.dask(where, open_files=open_files)
    read = _columns_read(monkeypatch)
    assert ak.array_equal(ev.rec.y.compute(), arr.rec.y)
    assert len(read) == 1
    read.clear()
    assert ak.array_equal(ev.rec.compute(), arr.rec)
    assert len(read) == 4  # x, y, and the offsets and content of lst
    # an unprojected read asks for top-level fields, which cover their subfields
    assert ak.array_equal(ev.compute(optimize_graph=False), arr)
    # a collection's offsets come with any field under it
    assert set(next(iter(dak.necessary_columns(ev.jets.eta).values()))) == {"jets.eta"}


@pytest.mark.parametrize("open_files", [True, False])
def test_a_collection_is_read_with_any_field_under_it(
    tmp_path, monkeypatch, open_files
):
    dask = pytest.importorskip("dask")
    dak = pytest.importorskip("dask_awkward")
    where, arr = _jets_with_hits(tmp_path)
    ev = uproot.dask(where, open_files=open_files)
    read = _columns_read(monkeypatch)
    assert ak.array_equal(ev.jets.pt.compute(), arr.jets.pt)
    assert len(read) == 2  # the jets offsets and pt, not hits
    read.clear()
    eta, num = dask.compute(ev.jets.eta, dak.num(ev.jets, axis=1))
    assert ak.array_equal(eta, arr.jets.eta)
    assert num.tolist() == ak.num(arr.jets, axis=1).tolist()
    assert len(read) == 2


@pytest.mark.parametrize("open_files", [True, False])
def test_an_empty_record_first_field_keeps_the_offsets(tmp_path, open_files):
    dak = pytest.importorskip("dask_awkward")
    where, arr = _rntuple(
        tmp_path,
        ak.Array([{"jets": [{"e": {}, "x": 1.0}] * (k % 3)} for k in range(6)]),
    )
    ev = uproot.dask(where, open_files=open_files)
    assert (
        dak.num(ev.jets, axis=1).compute().tolist() == ak.num(arr.jets, axis=1).tolist()
    )


def test_a_subfield_is_read_without_the_rest_of_its_collection(tmp_path):
    from uproot._dask import TrivialFormMappingInfo, _read_tree
    from uproot.source.futures import TrivialExecutor

    where, arr = _jets_with_hits(tmp_path)
    nt = uproot.open(where)
    expected = ak.forms.form_with_unique_keys(nt.to_akform()[0], ("<root>",))
    out = _read_tree(
        nt,
        {"jets.eta"},
        0,
        nt.num_entries,
        expected_form=expected,
        form_mapping_info=TrivialFormMappingInfo(expected, subfields=True),
        interp_options={"ak_add_doc": False},
        decompression_executor=TrivialExecutor(),
        interpretation_executor=TrivialExecutor(),
    )
    assert ak.array_equal(out.jets.eta, arr.jets.eta)


def test_graphed_reads_only_the_subfields_it_needs(tmp_path, monkeypatch):
    graphed = pytest.importorskip("graphed")
    from graphed.core.execution import SequentialRunner

    where, arr = _nested_rntuple(tmp_path)
    ev = uproot.graphed(where)
    assert ak.array_equal(ak.Array(ev.session.materialize(ev)), arr)
    assert uproot._graphed.necessary_columns(ev.rec.y) == {"nt": frozenset({"rec.y"})}
    plan = graphed.aggregate_plan(
        ev.rec.y,
        reduce=lambda vals: int(ak.sum(vals[0])),
        combine=lambda a, b: a + b,
        empty=lambda: 0,
    )
    read = _columns_read(monkeypatch)
    assert SequentialRunner().run(plan).value == ak.sum(arr.rec.y)
    assert len(read) == 1


def test_a_form_mapping_may_name_rntuple_subfields(tmp_path):
    dak = pytest.importorskip("dask_awkward")
    from uproot._dask import TrivialFormMappingInfo

    where, arr = _nested_rntuple(tmp_path)

    def mapping(form):
        expected = ak.forms.form_with_unique_keys(form, ("<root>",))
        return expected, TrivialFormMappingInfo(expected, subfields=True)

    # known_base_form opens no file, so only the mapping knows these are subfields
    ev = uproot.dask(
        where,
        open_files=False,
        known_base_form=uproot.open(where).to_akform()[0],
        form_mapping=mapping,
    )
    assert set(next(iter(dak.necessary_columns(ev.rec.x).values()))) == {"rec.x"}
    assert ak.array_equal(ev.rec.x.compute(), arr.rec.x)
    assert ak.array_equal(ev.compute(), arr)


def test_virtual_arrays_replace_unread_subfields(tmp_path):
    from uproot._dask import FormMappingInfoWithVirtualArrays, _read_tree
    from uproot.source.futures import TrivialExecutor

    where, arr = _nested_rntuple(tmp_path)
    nt = uproot.open(where)
    expected = ak.forms.form_with_unique_keys(nt.to_akform()[0], ("<root>",))
    out = _read_tree(
        nt,
        {"rec.x"},
        0,
        nt.num_entries,
        expected_form=expected,
        form_mapping_info=FormMappingInfoWithVirtualArrays(expected, subfields=True),
        interp_options={"ak_add_doc": False},
        decompression_executor=TrivialExecutor(),
        interpretation_executor=TrivialExecutor(),
    )
    assert ak.array_equal(ak.materialize(out.rec.y), arr.rec.y)
    assert ak.array_equal(ak.materialize(out.jets.pt), arr.jets.pt)


@pytest.fixture
def rntuple_and_ttree(tmp_path):
    ttree = skhep_testdata.data_path("uproot-HZZ-objects.root") + ":events"
    arr = uproot.open(ttree).arrays(["jetp4"], entry_stop=20)
    path = os.path.join(tmp_path, "jetp4.root")
    with uproot.recreate(path) as f:
        f.mkrntuple("events", arr)
    return [path + ":events", ttree]


def test_a_dataset_with_a_ttree_reads_whole_fields(rntuple_and_ttree):
    dak = pytest.importorskip("dask_awkward")
    ev = uproot.dask(rntuple_and_ttree, filter_name="jetp4")
    assert set(next(iter(dak.necessary_columns(ev.jetp4.fE).values()))) == {"jetp4"}
    fE = ev.jetp4.fE.compute()
    assert ak.array_equal(fE[:20], uproot.open(rntuple_and_ttree[0]).arrays().jetp4.fE)


def test_a_ttree_reads_the_fields_its_subfield_keys_are_under(rntuple_and_ttree):
    pytest.importorskip("dask_awkward")
    # without open_files, only the first file says whether the dataset is an RNTuple
    ev = uproot.dask(rntuple_and_ttree, filter_name="jetp4", open_files=False)
    fE = ev.jetp4.fE.compute()
    assert ak.array_equal(fE[20:], uproot.open(rntuple_and_ttree[1]).arrays().jetp4.fE)


def test_rntuple_buffers_are_keyed_by_their_own_node():
    from uproot._dask import TrivialFormMappingInfo

    form = ak.forms.form_with_unique_keys(
        ak.forms.from_dict(
            {
                "class": "RecordArray",
                "fields": ["jets", "u"],
                "contents": [
                    {
                        "class": "ListOffsetArray",
                        "offsets": "i64",
                        "content": {
                            "class": "RecordArray",
                            "fields": ["pt", "eta"],
                            "contents": [
                                {"class": "NumpyArray", "primitive": "float64"},
                                {"class": "NumpyArray", "primitive": "float64"},
                            ],
                        },
                    },
                    {
                        "class": "UnionArray",
                        "tags": "i8",
                        "index": "i64",
                        "contents": [
                            {"class": "NumpyArray", "primitive": "int64"},
                            {
                                "class": "RecordArray",
                                "fields": ["a"],
                                "contents": [
                                    {"class": "NumpyArray", "primitive": "bool"}
                                ],
                            },
                        ],
                    },
                ],
            }
        ),
        ("<root>",),
    )
    by_field = TrivialFormMappingInfo.build_form_key_to_key(form)
    by_node = TrivialFormMappingInfo.build_form_key_to_key(form, subfields=True)
    jets = form.content("jets")
    eta = jets.content.content("eta")
    assert by_field[jets.form_key] == by_field[eta.form_key] == "jets"
    assert by_node[jets.form_key] == "jets"
    assert by_node[eta.form_key] == "jets.eta"
    union = form.content("u")
    assert {by_node[c.form_key] for c in (union, *union.contents)} == {"u"}
    assert set(by_field.values()) == {None, "jets", "u"}

    buffers = frozenset({f"{jets.form_key}-offsets", f"{eta.form_key}-data"})
    assert TrivialFormMappingInfo(form).keys_for_buffer_keys(buffers) == {"jets"}
    assert TrivialFormMappingInfo(form, subfields=True).keys_for_buffer_keys(
        buffers
    ) == {"jets.eta"}


@pytest.mark.parametrize("name", ["[ab]", "{a,b}", "x*"])
def test_a_field_name_is_selected_literally(tmp_path, monkeypatch, name):
    pytest.importorskip("dask_awkward")
    where, arr = _rntuple(
        tmp_path,
        ak.Array(
            [{"rec": {name: k, "a": 2 * k, "b": 3 * k, "x1": 4 * k}} for k in range(4)]
        ),
    )
    ev = uproot.dask(where)
    read = _columns_read(monkeypatch)
    assert ak.array_equal(ev["rec", name].compute(), arr["rec", name])
    assert len(read) == 1


def test_a_top_level_field_name_may_contain_a_dot(tmp_path, monkeypatch):
    pytest.importorskip("dask_awkward")
    # ROOT forbids dots in RNTuple field names, but uproot can write them
    where, arr = _rntuple(
        tmp_path, ak.Array([{"a.b": k, "c": {"d": 2 * k}} for k in range(4)])
    )
    ev = uproot.dask(where)
    read = _columns_read(monkeypatch)
    assert ak.array_equal(ev["a.b"].compute(), arr["a.b"])
    assert len(read) == 1


def test_a_buffer_of_the_wrong_dtype_is_an_error(tmp_path):
    from uproot._dask import TrivialFormMappingInfo
    from uproot.source.futures import TrivialExecutor

    where, _ = _nested_rntuple(tmp_path)
    nt = uproot.open(where)
    form = nt.to_akform()[0].to_dict()
    form["contents"][form["fields"].index("z")]["primitive"] = "int32"
    expected = ak.forms.form_with_unique_keys(ak.forms.from_dict(form), ("<root>",))
    with pytest.raises(TypeError, match="Data type mismatch"):
        TrivialFormMappingInfo(expected, subfields=True).load_buffers(
            nt,
            frozenset({"z"}),
            0,
            nt.num_entries,
            TrivialExecutor(),
            TrivialExecutor(),
            {"ak_add_doc": False},
        )
