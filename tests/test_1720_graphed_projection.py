# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
"""Column projection: the ``TBranches`` a recorded graph needs are the ones read, and the
unread ones are placeholders the evaluation never touches."""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")

import graphed  # noqa: E402
from graphed.awkward import gak  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402

from tests.graphed_helpers import mini_nanoaod as mini  # noqa: E402

necessary_columns = uproot._graphed.necessary_columns


def _zmumu():
    return skhep_testdata.data_path("uproot-Zmumu.root") + ":events"


def _spy_reads(array):
    """Every (branches, start, stop) the source reads, in order."""
    _nid, source = uproot._graphed._uproot_source(array, "test")
    seen = []
    real = source.read_range
    source.read_range = lambda tree, keys, start, stop: (
        seen.append(tuple(keys)),
        real(tree, keys, start, stop),
    )[1]
    return seen


def test_necessary_columns_are_the_branches_the_graph_touches():
    g = uproot.graphed(_zmumu())
    assert necessary_columns(g.px1 + g.py1) == {"events": frozenset({"px1", "py1"})}
    assert necessary_columns(g[["px1"]]) == {"events": frozenset({"px1"})}
    assert necessary_columns(gak.num(g.px1, axis=0)) == {"events": frozenset()}


def test_a_plan_reads_exactly_the_projected_branches():
    g = uproot.graphed(_zmumu())
    out = g.px1 + g.py1
    seen = _spy_reads(out)
    plan = graphed.aggregate_plan(
        out,
        reduce=lambda vals: float(ak.sum(vals[0])),
        combine=lambda a, b: a + b,
        empty=lambda: 0.0,
        steps_per_file=2,
    )
    raw = uproot.open(_zmumu()).arrays(["px1", "py1"])
    assert SequentialRunner().run(plan).value == float(ak.sum(raw.px1 + raw.py1))
    assert seen == [("px1", "py1"), ("px1", "py1")]


def test_unread_branches_are_placeholders_the_graph_can_pass_through(tmp_path):
    # `zip` replays a read of `y`, then `[["a"]]` drops it: `y` is projected away and its
    # placeholder buffer is never touched
    g = uproot.graphed(_zmumu())
    rec = gak.zip({"a": g.px1, "b": g.py1})[["a"]]
    assert necessary_columns(rec) == {"events": frozenset({"px1"})}
    assert uproot.graphed_head(rec, 2).fields == ["a"]
    paths = uproot.graphed_write(rec, os.path.join(tmp_path, "out"), executor="thread")
    assert uproot.open(paths[0])["tree"].keys() == ["a"]
    # the same through a restructuring mapping: a sibling collection's data stays unread
    events = uproot.graphed(
        _nano(),
        filter_name=mini.FILTER,
        form_mapping=mini.MiniNanoMapping(),
        session=graphed.Session(graphed.awkward.AwkwardBackend(behavior=mini.BEHAVIOR)),
    )
    z = gak.zip({"jet": events.Jet.pt, "muon": events.Muon.pt}, depth_limit=1)[["jet"]]
    assert necessary_columns(z) == {"Events": frozenset({"nJet", "Jet_pt", "nMuon"})}
    assert uproot.graphed_head(z, 1).fields == ["jet"]


def _nano():
    return skhep_testdata.data_path("nanoAOD_2015_CMS_Open_Data_ttbar.root") + ":Events"


def test_an_opaque_node_reads_everything_its_input_carries():
    g = uproot.graphed(_zmumu(), filter_name=["px1", "py1", "pz1"])
    doubled = graphed.apply(lambda x: x * 2, g.px1)
    assert necessary_columns(doubled, on_fail="pass") == {"events": frozenset({"px1"})}
    whole = graphed.apply(lambda ev: ev.px1, g)
    assert necessary_columns(whole, on_fail="pass") == {
        "events": frozenset({"px1", "py1", "pz1"})
    }
    assert uproot.graphed_head(doubled, 2).tolist() == pytest.approx(
        (2 * uproot.open(_zmumu())["px1"].array(entry_stop=2)).tolist()
    )


@pytest.fixture
def kinematics(tmp_path):
    vector = pytest.importorskip("vector")
    vector.register_awkward()
    rng = np.random.default_rng(7)
    cols = {
        "px": rng.normal(0.0, 10.0, 50),
        "py": rng.normal(0.0, 10.0, 50),
        "pz": rng.normal(0.0, 20.0, 50),
        "E": rng.uniform(30.0, 60.0, 50),
        "unused": rng.normal(size=50),
    }
    path = os.path.join(tmp_path, "kin.root")
    with uproot.recreate(path) as f:
        f.mktree("events", {k: "f8" for k in cols})
        f["events"].extend(cols)
    reference = ak.Array(
        ak.zip(
            {k: cols[k] for k in ("px", "py", "pz", "E")}, with_name="Momentum4D"
        ).layout,
        behavior=vector.backends.awkward.behavior,
    )
    return path + ":events", vector.backends.awkward.behavior, reference


def test_behavior_properties_project_to_their_branches_and_evaluate(kinematics):
    where, behavior, reference = kinematics
    g = uproot.graphed(where, behavior=behavior)
    v = gak.with_name(
        gak.zip({"px": g.px, "py": g.py, "pz": g.pz, "E": g.E}), "Momentum4D"
    )
    assert necessary_columns(v.pt) == {"events": frozenset({"px", "py"})}
    assert necessary_columns(v.mass) == {"events": frozenset({"px", "py", "pz", "E"})}
    assert ak.array_equal(uproot.graphed_head(v.pt, 50), reference.pt)
    with pytest.raises(graphed.GraphedTypeError):
        v.no_such_property  # noqa: B018  (an unknown attribute fails at record time)


def test_to_parquet_reads_only_the_declared_branches(tmp_path):
    pytest.importorskip("pyarrow")
    g = uproot.graphed(_zmumu())
    out = gak.zip({"px": g.px1})
    seen = _spy_reads(out)
    from graphed.awkward import to_parquet

    paths = to_parquet(out, os.path.join(tmp_path, "pq"), steps_per_file=2)
    assert len(paths) == 2 and seen == [("px1",), ("px1",)]
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    assert ak.array_equal(back.px, uproot.open(_zmumu())["px1"].array())
