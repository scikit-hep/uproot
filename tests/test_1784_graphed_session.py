# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""The Session ``uproot.graphed`` records into: ``session=``, the current Session, ``name=``."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import os

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")

import graphed  # noqa: E402
from graphed.awkward import AwkwardBackend, gak  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402

from tests.graphed_helpers import mini_nanoaod as mini  # noqa: E402


def _zmumu():
    return skhep_testdata.data_path("uproot-Zmumu.root") + ":events"


def _nano():
    return skhep_testdata.data_path("nanoAOD_2015_CMS_Open_Data_ttbar.root") + ":Events"


def _keyed_tree(path, field, values):
    with uproot.recreate(path) as f:
        f["events"] = {
            "k": np.arange(len(values), dtype="i8"),
            field: np.asarray(values, "f8"),
        }
    return path + ":events"


def _joined(tmp_path):
    left = uproot.graphed(
        _keyed_tree(os.path.join(tmp_path, "a.root"), "x", [1.0, 2.0, 3.0])
    )
    right = uproot.graphed(
        _keyed_tree(os.path.join(tmp_path, "b.root"), "z", [10.0, 20.0]),
        session=left.session,
        name="friends",
    )
    return left, right, graphed.join(left, right, on=["k"])


def test_session_and_backend_are_exclusive():
    with pytest.raises(TypeError, match="pass backend= or session=, not both"):
        uproot.graphed(
            _zmumu(),
            backend=AwkwardBackend(),
            session=graphed.Session(AwkwardBackend()),
        )
    with pytest.raises(TypeError, match="behavior= is for the default backend"):
        uproot.graphed(_zmumu(), behavior={}, session=graphed.Session(AwkwardBackend()))


def test_a_read_into_another_reads_session_joins_with_it(tmp_path):
    left, right, joined = _joined(tmp_path)
    session = left.session
    assert right.session is session
    assert [session.source_name(nid) for nid in session.source_ids()] == [
        "events",
        "friends",
    ]
    assert session.form(right).describe() == "## * {k: int64, z: float64}"

    plan = graphed.join_plan(
        joined.x + joined.z,
        reduce=lambda vals: sorted(ak.to_list(vals[0])),
        combine=lambda a, b: sorted(a + b),
        empty=list,
    )
    assert SequentialRunner().run(plan).value == [11.0, 22.0]
    assert uproot.graphed_head(right.z, 5).tolist() == [10.0, 20.0]


def test_session_must_be_an_awkward_session():
    with pytest.raises(TypeError, match="takes a graphed.Session, not AwkwardBackend"):
        uproot.graphed(_zmumu(), session=AwkwardBackend())
    from graphed.numpy import NumpyBackend

    with pytest.raises(
        TypeError, match="records on NumpyBackend, and this read on AwkwardBackend"
    ):
        uproot.graphed(_zmumu(), session=graphed.Session(NumpyBackend()))


def test_form_mapping_needs_the_sessions_behavior():
    def read(session, name, **kwargs):
        return uproot.graphed(
            _nano(),
            filter_name=mini.FILTER,
            form_mapping=mini.MiniNanoMapping(**kwargs),
            session=session,
            name=name,
        )

    with pytest.raises(
        TypeError,
        match="records on AwkwardBackend with another behavior, and this read on "
        "AwkwardBackend with its form_mapping behavior",
    ):
        read(graphed.Session(AwkwardBackend()), "nano")
    mapped = uproot.graphed(
        _nano(), filter_name=mini.FILTER, form_mapping=mini.MiniNanoMapping()
    )
    assert read(mapped.session, "nano").session is mapped.session
    with pytest.raises(TypeError, match="with another behavior"):
        read(mapped.session, "other", behavior={})


def test_without_current_session_each_read_is_new_and_a_taken_name_is_refused(
    monkeypatch,
):
    monkeypatch.delattr(graphed, "current_session", raising=False)
    path = _zmumu()
    session = uproot.graphed(path).session
    assert uproot.graphed(path).session is not session
    with pytest.raises(ValueError, match="'events'; pass name= to name this read"):
        uproot.graphed(path.rpartition(":")[0], session=session)  # named by the TTree
    mine = uproot.graphed(path, name="mine")
    assert [mine.session.source_name(nid) for nid in mine.session.source_ids()] == [
        "mine"
    ]


def test_a_known_taken_name_is_refused_before_any_file_opens(tmp_path, monkeypatch):
    monkeypatch.delattr(graphed, "current_session", raising=False)
    session = uproot.graphed(_zmumu()).session
    missing = os.path.join(tmp_path, "missing.root")
    for files, name in ((missing + ":events", None), (missing, "events")):
        with pytest.raises(ValueError, match="'events'; pass name= to name this read"):
            uproot.graphed(files, session=session, name=name)


def test_a_branch_path_is_not_refused_under_the_branch_name():
    session = uproot.graphed(_zmumu(), name="px1").session
    leaf = uproot.graphed(_zmumu() + "/px1", session=session)
    assert [session.source_name(nid) for nid in session.source_ids()] == [
        "px1",
        "events",
    ]
    assert leaf.session is session


def test_head_finds_its_source_by_node_not_name(tmp_path):
    left, _, _ = _joined(tmp_path)
    session = left.session
    if hasattr(graphed, "current_session"):  # one binding per name
        with pytest.raises(ValueError, match="'events' is already bound to other data"):
            session.source("events", form=session.form(left), data=None, tag=1)
        return
    # another integration's source under the same name, kept apart by its params
    other = session.source("events", form=session.form(left), data=None, tag=1)
    assert len(session.source_ids()) == 3
    with pytest.raises(TypeError, match="this array is backed by 0"):
        uproot.graphed_head(other.x)


def test_necessary_columns_projects_once_over_the_arrays_own_sources(
    tmp_path, monkeypatch
):
    left, _, joined = _joined(tmp_path)
    assert uproot._graphed.necessary_columns(left.x) == {"events": frozenset({"x"})}

    calls = []
    project = graphed.awkward.projection.project_buffers_many

    def counting(*args, **kwargs):
        calls.append(args)
        return project(*args, **kwargs)

    monkeypatch.setattr(graphed.awkward.projection, "project_buffers_many", counting)
    assert uproot._graphed.necessary_columns(joined.x + joined.z) == {
        "events": frozenset({"k", "x"}),
        "friends": frozenset({"k", "z"}),
    }
    assert len(calls) == 1


def test_head_and_write_count_sources_before_compiling(tmp_path, monkeypatch):
    _, _, joined = _joined(tmp_path)

    def no_compile(*args, **kwargs):
        raise AssertionError("compiled before counting sources")

    monkeypatch.setattr(graphed, "compile_ir", no_compile)
    with pytest.raises(TypeError, match="graphed_head .* backed by 2"):
        uproot.graphed_head(joined.x + joined.z)
    with pytest.raises(TypeError, match="graphed_write .* backed by 2"):
        uproot.graphed_write(joined, os.path.join(tmp_path, "out"))


needs_current_session = pytest.mark.skipif(
    not hasattr(graphed, "current_session"), reason="graphed has no current Session"
)


class _NanoLikeBackend(AwkwardBackend):
    """A behavior-carrying ``AwkwardBackend`` subclass, as coffea's ``GraphedNanoBackend``."""


@dataclasses.dataclass
class _EqualExecutor:
    """An executor whose ``==`` compares settings, so only ``is`` tells two apart."""

    workers: int = 1

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        future.set_result(fn(*args, **kwargs))
        return future


class _IdentifiedMapping(mini.MiniNanoMapping):
    def source_identity(self):
        return "mini"


def _names(session):
    return [session.source_name(nid) for nid in session.source_ids()]


@needs_current_session
def test_a_read_outside_any_scope_records_into_the_default(tmp_path):
    left = uproot.graphed(
        _keyed_tree(os.path.join(tmp_path, "a.root"), "x", [1.0, 2.0, 3.0])
    )
    assert left.session is graphed.current_session()
    assert type(left.session.backend) is AwkwardBackend
    right = uproot.graphed(
        _keyed_tree(os.path.join(tmp_path, "b.root"), "z", [10.0, 20.0]),
        name="friends",
    )
    assert right.session is left.session
    joined = graphed.join(left, right, on=["k"])
    assert ak.to_list(left.session.materialize(joined.x + joined.z)) == [11.0, 22.0]


@needs_current_session
def test_a_behavior_read_creates_the_default_on_its_behavior():
    first = uproot.graphed(_zmumu(), behavior=mini.BEHAVIOR)
    assert first.session is graphed.current_session()
    assert first.session.backend._behavior == mini.BEHAVIOR
    again = uproot.graphed(_zmumu(), behavior=dict(mini.BEHAVIOR), name="again")
    assert again.session is first.session
    with pytest.raises(
        TypeError,
        match="the current Session records on AwkwardBackend with another behavior, and "
        "this read on AwkwardBackend with the behavior= given",
    ):
        uproot.graphed(_zmumu(), behavior={("*", "Other"): mini.MiniJet}, name="c")


@needs_current_session
def test_an_empty_behavior_lands_in_the_default_it_created():
    assert uproot.graphed(_zmumu(), behavior={}).session is graphed.current_session()


@needs_current_session
def test_an_equal_backend_lands_in_the_default():
    default = uproot.graphed(_zmumu()).session
    assert uproot.graphed(_zmumu(), backend=AwkwardBackend(), name="b").session is (
        default
    )


@needs_current_session
def test_a_default_on_another_backend_is_refused_at_the_read():
    from graphed.numpy import NumpyBackend

    refused = (
        "the current Session records on NumpyBackend, and this read on AwkwardBackend; "
        "pass a Session on that backend with session=, call graphed.new_session"
    )
    graphed.new_session(NumpyBackend())
    with pytest.raises(TypeError, match=refused):
        uproot.graphed(_zmumu())
    graphed.new_session(None)
    with graphed.Session(NumpyBackend()), pytest.raises(TypeError, match=refused):
        uproot.graphed(_zmumu())


@needs_current_session
def test_reads_in_a_scope_record_there(tmp_path):
    default = graphed.current_session()
    with graphed.Session(AwkwardBackend()) as scope:
        a = uproot.graphed(_zmumu())
        b = uproot.graphed(_zmumu(), name="b")
        assert a.session is b.session is scope
        assert (
            uproot.graphed(_zmumu(), backend=scope.backend, name="c").session is scope
        )
        assert uproot.graphed(_zmumu(), backend=AwkwardBackend(), name="d").session is (
            scope
        )
        for backend in (
            _NanoLikeBackend(),
            AwkwardBackend(behavior=mini.BEHAVIOR),
        ):
            with pytest.raises(TypeError, match="the current Session records on"):
                uproot.graphed(_zmumu(), backend=backend, name="e")
        other = graphed.Session(AwkwardBackend())
        assert uproot.graphed(_zmumu(), session=other).session is other
        assert _names(scope) == ["events", "b", "c", "d"]
    assert _names(default) == []


@needs_current_session
@pytest.mark.parametrize(
    "backend", [AwkwardBackend(), _NanoLikeBackend(behavior=mini.BEHAVIOR)]
)
def test_a_plain_read_joins_inside_the_scope_of_its_partner(tmp_path, backend):
    events = uproot.graphed(
        _keyed_tree(os.path.join(tmp_path, "a.root"), "x", [1.0, 2.0, 3.0]),
        backend=backend,
    )
    with events.session:
        friends = uproot.graphed(
            _keyed_tree(os.path.join(tmp_path, "b.root"), "z", [10.0, 20.0]),
            name="friends",
        )
    assert friends.session is events.session
    joined = graphed.join(events, friends, on=["k"])
    assert ak.to_list(events.session.materialize(joined.z)) == [10.0, 20.0]


@needs_current_session
def test_the_same_read_twice_is_one_source():
    first = uproot.graphed(_zmumu())
    nodes = first.session.node_count()
    again = uproot.graphed(_zmumu())
    assert again.session is first.session
    assert again.node_id == first.node_id
    assert first.session.node_count() == nodes


@needs_current_session
def test_a_mapped_read_is_refused_by_a_default_without_its_behavior():
    uproot.graphed(_zmumu())
    with pytest.raises(
        TypeError,
        match="the current Session records on AwkwardBackend with another behavior, and "
        "this read on AwkwardBackend with its form_mapping behavior",
    ):
        uproot.graphed(
            _nano(), filter_name=mini.FILTER, form_mapping=mini.MiniNanoMapping()
        )


@needs_current_session
@pytest.mark.parametrize(
    "changed",
    [
        {"files": "other"},
        {"steps": [[0, 1], [1, 2]]},
        {"allow_missing": True},
        {"align_baskets": True},
        {"custom_classes": dict(uproot.classes)},
    ],
    ids=lambda changed: next(iter(changed)),
)
def test_a_name_reread_with_anything_else_is_refused(tmp_path, changed):
    a, b = (os.path.join(tmp_path, f"{x}.root") for x in "ab")
    for path in (a, b):
        with uproot.recreate(path) as f:
            f.mktree("events", {"x": "f8"})
            f["events"].extend({"x": np.arange(2.0)})
    a, b = a + ":events", b + ":events"
    uproot.graphed(a)
    kwargs = dict(changed)
    files = b if kwargs.pop("files", None) else a
    if "steps" in kwargs:
        files = {
            a.rpartition(":")[0]: {
                "object_path": "events",
                "steps": kwargs.pop("steps"),
            }
        }
    with pytest.raises(ValueError, match="'events' is already bound to other data"):
        uproot.graphed(files, name="events", **kwargs)


@needs_current_session
def test_a_mapping_is_its_own_identity_unless_it_defines_one():
    def read(mapping):
        return uproot.graphed(
            _nano(), filter_name=mini.FILTER, form_mapping=mapping, name="Events"
        )

    mapping = mini.MiniNanoMapping()
    assert read(mapping).node_id == read(mapping).node_id
    with pytest.raises(ValueError, match="'Events' is already bound to other data"):
        read(mini.MiniNanoMapping())
    graphed.new_session(None)
    assert read(_IdentifiedMapping()).node_id == read(_IdentifiedMapping()).node_id


@needs_current_session
def test_executors_are_compared_by_identity():
    executor = _EqualExecutor()
    first = uproot.graphed(_zmumu(), decompression_executor=executor)
    assert uproot.graphed(_zmumu(), decompression_executor=executor).node_id == (
        first.node_id
    )
    with pytest.raises(ValueError, match="'events' is already bound to other data"):
        uproot.graphed(_zmumu(), decompression_executor=_EqualExecutor(), name="events")
    with pytest.raises(ValueError, match="'events' is already bound to other data"):
        uproot.graphed(
            _zmumu(), interpretation_executor=_EqualExecutor(), name="events"
        )


@needs_current_session
def test_without_current_session_a_read_gets_a_new_session(monkeypatch):
    default = graphed.current_session()
    monkeypatch.delattr(graphed, "current_session")
    assert uproot.graphed(_zmumu()).session is not default
    assert _names(default) == []


@pytest.mark.skipif(
    not hasattr(graphed.aggregate, "worker_backend"),
    reason="graphed has no worker_backend",
)
def test_workers_evaluate_with_the_sessions_behavior(tmp_path):
    g = uproot.graphed(
        _keyed_tree(os.path.join(tmp_path, "a.root"), "x", [1.0, 2.0, 3.0]),
        behavior=_DOUBLING,
    )
    out = gak.zip({"d": gak.with_name(gak.zip({"x": g.x}), "Doubling").doubled()})
    paths = uproot.graphed_write(out, os.path.join(tmp_path, "out"), executor="thread")
    assert uproot.open(paths[0])["tree"]["d"].array().tolist() == [2.0, 4.0, 6.0]


class _Doubling(ak.Array):
    def doubled(self):
        return self.x * 2


_DOUBLING = {("*", "Doubling"): _Doubling}
