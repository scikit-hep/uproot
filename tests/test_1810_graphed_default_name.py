# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Default-named ``uproot.graphed`` reads of different data share a ``Session``."""

from __future__ import annotations

import os

import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")

import graphed  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(graphed, "current_session"), reason="graphed has no current Session"
)


def _zmumu():
    return skhep_testdata.data_path("uproot-Zmumu.root") + ":events"


def _names(session):
    return [session.source_name(nid) for nid in session.source_ids()]


def _executor():
    return uproot.source.futures.TrivialExecutor()


def _file(tmp_path, stem, *trees, offset=0.0):
    path = os.path.join(tmp_path, f"{stem}.root")
    with uproot.recreate(path) as f:
        for tree in trees:
            f.mktree(tree, {"x": "f8", "y": "f8"}).extend(
                {"x": np.arange(3.0) + offset, "y": np.arange(3.0)}
            )
    return path


def _events(tmp_path, stem, offset=0.0):
    return _file(tmp_path, stem, "Events", offset=offset) + ":Events"


def _ir(array):
    return bytes(graphed.compile_ir(array.session, array).ir)


_BASE = {"p": "Events", "o": "A"}
_STEPS = {
    "p": {"object_path": "Events", "steps": [[0, 1]]},
    "o": {"object_path": "A", "steps": [[0, 1]]},
}


def test_default_reads_of_different_data_get_sources_a_reread_interns(tmp_path):
    dy = _events(tmp_path, "dy")
    reads = [
        (dy, {}),
        (_events(tmp_path, "tt", 100.0), {}),
        (dy, {"filter_name": "x"}),
    ]
    whole, other, only_x = arrays = [
        uproot.graphed(path, **options) for path, options in reads
    ]
    session = whole.session
    again = [uproot.graphed(path, **options) for path, options in reads]
    assert [a.node_id for a in again] == [a.node_id for a in arrays]
    assert session.node_count() == 3
    first, *rest = _names(session)
    assert first == "Events"
    assert all(name.startswith("Events-") for name in rest)
    assert uproot._graphed.necessary_columns(other.x) == {rest[0]: frozenset({"x"})}
    assert only_x.fields == ["x"]
    assert graphed.materialize(other.x).tolist() == [100.0, 101.0, 102.0]
    assert graphed.materialize(whole.y).tolist() == [0.0, 1.0, 2.0]
    assert graphed.materialize(only_x.x).tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("tree", ["Events", "a/Events"])
def test_every_spelling_of_a_tree_interns(tmp_path, tree):
    path = _file(tmp_path, "dy", tree)
    tt = _events(tmp_path, "tt")
    uproot.graphed(tt)
    first = uproot.graphed(f"{path}:{tree}")
    later = {tt.rpartition(":")[0]: "Events", path: f"{tree};1"}
    assert (
        uproot.graphed(later).node_id == uproot.graphed(later, open_files=True).node_id
    )
    for flag in ("allow_missing", "align_baskets"):
        one = uproot.graphed(f"{path}:{tree}", **{flag: 1})
        assert one.node_id == uproot.graphed(path, **{flag: True}).node_id
    spellings = [path, f"{path}:{tree};1", f"{path}:/{tree}", {path: f"/{tree};1"}]
    with uproot.open(path) as f:
        spellings.append(f[tree])
        assert {uproot.graphed(s).node_id for s in spellings} == {first.node_id}


def test_a_later_read_shifts_no_name_and_keeps_the_first_ir(tmp_path):
    dy, tt, ww = (_events(tmp_path, stem) for stem in ("dy", "tt", "ww"))
    alone = _ir(uproot.graphed(dy).x)
    graphed.new_session(None)
    first = uproot.graphed(dy)
    uproot.graphed(tt)
    three = _names(uproot.graphed(ww).session)
    assert _ir(first.x) == alone
    graphed.new_session(None)
    uproot.graphed(dy)
    two = _names(uproot.graphed(ww).session)
    assert two == [three[0], three[2]]


def test_reads_the_digest_cannot_tell_apart_are_numbered():
    executors = [_executor() for _ in range(3)]
    arrays = [
        uproot.graphed(_zmumu(), decompression_executor=executor)
        for executor in executors
    ]
    last = uproot.graphed(_zmumu(), custom_classes=dict(uproot.classes))
    first, hashed, *numbered = _names(last.session)
    assert first == "events"
    assert hashed.startswith("events-")
    assert numbered == [f"{hashed}#1", f"{hashed}#2"]
    assert (
        uproot.graphed(_zmumu(), decompression_executor=executors[2]).node_id
        == arrays[2].node_id
    )


def test_names_lost_to_another_thread_move_on(monkeypatch):
    source = graphed.Session.source
    rivals = [_executor(), _executor()]

    def racing(self, name, *, form, data):
        while rivals:
            uproot.graphed(_zmumu(), decompression_executor=rivals.pop())
        return source(self, name, form=form, data=data)

    monkeypatch.setattr(graphed.Session, "source", racing)
    array = uproot.graphed(_zmumu())
    _, hashed, numbered = _names(array.session)
    assert numbered == f"{hashed}#1"
    assert array.session.source_name(array.node_id) == numbered


def test_candidates_stop_at_one_more_than_the_sources(monkeypatch):
    session = uproot.graphed(_zmumu()).session
    uproot.graphed(_zmumu(), decompression_executor=_executor())
    first, hashed = _names(session)
    candidates = [first, hashed, f"{hashed}#1"]
    tried = []

    def refuse(self, name, *, form, data):
        tried.append(name)
        raise ValueError(name)

    monkeypatch.setattr(graphed.Session, "source", refuse)
    with pytest.raises(ValueError, match="#1"):
        uproot.graphed(_zmumu(), decompression_executor=_executor())
    assert tried == candidates

    tried.clear()
    monkeypatch.setattr(uproot._graphed, "_taken", lambda session, name: True)
    with pytest.raises(AssertionError, match="no free source name"):
        uproot.graphed(_zmumu(), decompression_executor=_executor())
    assert tried == candidates


@pytest.mark.parametrize(
    ("spec", "options"),
    [
        pytest.param({"p": "Events", "w": "A"}, {}, id="file"),
        pytest.param({"p": "a/Events", "o": "A"}, {}, id="directory"),
        pytest.param({"p": "Events", "o": "B"}, {}, id="later-tree"),
        pytest.param(_STEPS, {}, id="steps"),
        pytest.param(_BASE, {"allow_missing": True}, id="allow_missing"),
        pytest.param(_BASE, {"align_baskets": True}, id="align_baskets"),
        pytest.param(_BASE, {"filter_name": "x"}, id="form"),
    ],
)
def test_each_digest_part_tells_reads_apart(tmp_path, spec, options):
    paths = {
        "p": _file(tmp_path, "p", "Events", "a/Events"),
        "o": _file(tmp_path, "o", "A", "B"),
        "w": _file(tmp_path, "w", "A"),
    }

    def files(roles):
        return {paths[role]: entry for role, entry in roles.items()}

    uproot.graphed(_events(tmp_path, "bare"))
    uproot.graphed(files(_BASE))
    names = _names(uproot.graphed(files(spec), **options).session)
    assert len(names) == 3
    assert not any("#" in name for name in names)
