# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
"""``uproot.graphed(form_mapping=, known_base_form=, backend=)``: the schema'd source, through the
same mapping objects ``uproot.dask`` takes.

The mapping is ``tests.graphed_helpers.mini_nanoaod`` — a real restructuring (``nJet`` + ``Jet_*``
-> ``Jet: var * MiniJet[...]``), because ``TrivialFormMapping`` cannot tell graph field names and
branch names apart.
"""

from __future__ import annotations

import awkward as ak
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")

import graphed  # noqa: E402
from graphed.awkward import AwkwardBackend, gak  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402

from tests.graphed_helpers import mini_nanoaod as mini  # noqa: E402

MAPPED_FORM = (
    "## * {Jet: var * MiniJet[eta: float32, pt: float32], "
    "Muon: var * MiniJet[pt: float32], MET_pt: float32}"
)


def _nano():
    return skhep_testdata.data_path("nanoAOD_2015_CMS_Open_Data_ttbar.root") + ":Events"


def _mapped(**kwargs):
    return uproot.graphed(
        _nano(), filter_name=mini.FILTER, form_mapping=mini.MiniNanoMapping(), **kwargs
    )


def _source_of(array):
    _nid, src = uproot._graphed._uproot_source(array, "test")
    return src


def _total(out, **kwargs):
    plan = graphed.aggregate_plan(
        out,
        reduce=lambda vals: float(ak.sum(vals[0])),
        combine=lambda a, b: a + b,
        empty=lambda: 0.0,
        **kwargs,
    )
    return SequentialRunner().run(plan).value


def test_mapped_form_and_behavior_match_uproot_dask():
    pytest.importorskip("dask_awkward")
    mapping = mini.MiniNanoMapping()
    g = uproot.graphed(_nano(), filter_name=mini.FILTER, form_mapping=mapping)
    d = uproot.dask(
        _nano(), filter_name=mini.FILTER, form_mapping=mapping, open_files=False
    )
    assert g.session.form(g).describe() == MAPPED_FORM == str(d._meta.type)
    # the mapping info's behavior reached the recording backend: `pt2` is a behavior PROPERTY
    assert g.session.form(g.Jet.pt2).describe() == "## * var * float32"
    got = ak.Array(g.session.materialize(g.Jet.pt2))
    assert ak.array_equal(got, mini.eager_events(_nano()).Jet.pt2)


def test_mapped_analysis_runs_partitioned_bit_for_bit():
    g = uproot.graphed(
        [_nano(), _nano()], filter_name=mini.FILTER, form_mapping=mini.MiniNanoMapping()
    )
    events = mini.eager_events(_nano(), n_copies=2)
    assert _total(gak.num(g.Jet, axis=1), steps_per_file=3) == float(
        ak.sum(ak.num(events.Jet, axis=1))
    )
    # the mapping's behavior rides on every chunk, so a worker backend with no behavior of its
    # own still resolves the property
    assert _total(gak.sum(g.Jet.pt2, axis=1), steps_per_file=3) == pytest.approx(
        float(ak.sum(events.Jet.pt2)), rel=1e-6
    )  # float32 partials: the partition order differs from one eager sum


def test_count_only_output_declares_the_counter_branch():
    g = _mapped()
    n_jet = gak.num(g.Jet, axis=1)
    assert uproot._graphed.necessary_columns(n_jet) == {"Events": frozenset({"nJet"})}
    assert _total(n_jet, steps_per_file=2) == float(
        ak.sum(ak.num(mini.eager_events(_nano()).Jet, axis=1))
    )


def test_known_base_form_records_without_opening_a_file(monkeypatch):
    base_form = _source_of(_mapped())._form_mapping_info.seen_base_form
    monkeypatch.setattr(
        uproot.reading.ReadOnlyFile,
        "__init__",
        lambda *a, **k: pytest.fail("a file was opened"),
    )
    g = uproot.graphed(
        {_nano().rsplit(":", 1)[0]: "Events"},
        form_mapping=mini.MiniNanoMapping(),
        known_base_form=base_form,
        session=graphed.Session(AwkwardBackend(behavior=mini.BEHAVIOR)),
    )
    assert g.session.form(g).describe() == MAPPED_FORM
    assert g.session.source_name(g.node_id) == "Events"


def test_backend_instance_installs_its_array_type():
    class MyBackend(AwkwardBackend):
        pass

    g = _mapped(backend=MyBackend(behavior=mini.BEHAVIOR))
    assert type(g.session.backend) is MyBackend
    with pytest.raises(TypeError, match="behavior= is for the default backend"):
        _mapped(behavior={})


def test_dict_ak_add_doc_and_typenames_reach_the_mapping():
    g = _mapped(ak_add_doc={"__doc__": "title", "typename": "typename"})
    seen = _source_of(g)._form_mapping_info.seen_base_form
    assert seen.parameters["typenames"]["nJet"] == "uint32_t"
    assert seen.content("nJet").parameters == {
        "__doc__": uproot.open(_nano())["nJet"].title,
        "typename": "uint32_t",
    }


def test_a_virtual_array_mapping_fills_unread_buffers_lazily(tmp_path, monkeypatch):
    from uproot._dask import (
        FormMappingInfoWithVirtualArrays,
        FormMappingWithVirtualArrays,
    )

    calls = []
    real = FormMappingInfoWithVirtualArrays.buffer_replacements

    def spy(self, *args, **kwargs):
        calls.append(args[1])  # the keys the replacement stands in for
        return real(self, *args, **kwargs)

    monkeypatch.setattr(FormMappingInfoWithVirtualArrays, "buffer_replacements", spy)
    where = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g = uproot.graphed(
        where, filter_name=["px1", "py1"], form_mapping=FormMappingWithVirtualArrays()
    )
    assert uproot.graphed_head(g.px1, 2).tolist() == pytest.approx(
        uproot.open(where)["px1"].array(entry_stop=2).tolist()
    )
    assert (
        frozenset({"py1"}) in calls
    )  # py1 was not read: its buffers are virtual, not placeholders
