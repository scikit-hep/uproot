# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest
import skhep_testdata

import uproot

graphed = pytest.importorskip("graphed")
pytest.importorskip("graphed.awkward")
from graphed.awkward import AwkwardBackend  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(graphed, "current_session"),
    reason="graphed has no one-binding-per-name rule",
)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """Every ``files`` spelling the tests name: ``cycles.root`` holds ``events;1`` (3 entries),
    ``events;2`` (5), ``other``, ``dir/events``, ``weird;name`` and ``dir;name/events``;
    ``single.root`` and ``single2.root`` hold only ``events``; ``<...>`` are already-open
    objects, their files closed but for ``<tree>.evt``'s, whose streamers are read lazily.
    """
    tmp = tmp_path_factory.mktemp("identity")
    cycles, single, single2 = (
        str(tmp / f"{n}.root") for n in ("cycles", "single", "single2")
    )
    with uproot.recreate(cycles) as file:
        for name in ("events", "other", "dir/events", "weird;name", "dir;name/events"):
            file.mktree(name, {"x": np.int32}).extend(
                {"x": np.arange(3, dtype=np.int32)}
            )
    with uproot.update(cycles) as file:
        file.mktree("events", {"x": np.int32}).extend(
            {"x": np.arange(5, dtype=np.int32)}
        )
    for path in (single, single2):
        with uproot.recreate(path) as file:
            file.mktree("events", {"x": np.int32}).extend(
                {"x": np.arange(3, dtype=np.int32)}
            )
    evnt = skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    with uproot.open(cycles, object_cache=None) as file:
        objects = {
            "<events>": file["events"],
            "<events;1>": file["events;1"],
            "<events>.x": file["events"]["x"],
            "<dir/events>": file["dir/events;1"],
        }
    with uproot.open(evnt, object_cache=None) as evnt_file:
        yield {
            **objects,
            "<tree>.evt": evnt_file["tree"]["evt"],
            "evnt:tree/evt": {evnt: "tree/evt"},
            "single": single,
            "single:events": {single: "events"},
            "pair": [single, single2],
            "mixed": {single: "events", cycles: "events;2"},
        }, cycles


def _read(inputs, spec, session):
    """``spec`` is a key of ``inputs`` (else an object path in ``cycles.root``), with ``|k`` for
    a ``known_base_form`` read and ``|o`` for ``open_files=True``."""
    named, cycles = inputs
    key, _, how = spec.partition("|")
    files = named.get(key, {cycles: key})

    def read(session, **options):
        return uproot.graphed(
            files,
            session=session,
            name="events",
            open_files=how == "o",
            **options,
        )

    if how != "k":
        return read(session)
    plain = read(graphed.Session(AwkwardBackend()))
    return read(session, known_base_form=plain.session.form(plain).tt.layout.form)


@pytest.mark.parametrize(
    "first, second",
    [
        ("events", "events|k"),
        ("events", "/events|k"),
        ("events;1", "events;1|k"),
        ("events", "events;2|k"),  # the highest cycle, explicit
        ("events;2", "events;2|k"),
        ("events", "events;2"),
        ("<events>", "events|k"),
        ("<events>|k", "events"),
        ("events/x", "events/x|k"),  # a leaf TBranch
        ("events:x", "events:x|k"),
        ("<events>.x|k", "events"),
        ("<events>.x", "events/x"),
        ("weird;name", "weird;name|k"),
        ("dir;name/events", "dir;name/events|k"),
        ("<dir/events>", "dir/events|k"),  # an open TTree, its file closed
        ("<dir/events>|k", "dir/events"),
        ("evnt:tree/evt", "<tree>.evt"),  # a non-leaf TBranch
        ("<tree>.evt|k", "evnt:tree/evt"),
        ("single", "single|k"),  # no object path
        ("single", "single:events"),
        ("pair", "pair|o"),  # a later file opened or not
        ("mixed", "mixed|o"),
    ],
)
@pytest.mark.parametrize("swap", [False, True])
def test_spellings_of_one_tree_share_a_source(inputs, first, second, swap):
    if swap:
        first, second = second, first
    session = graphed.Session(AwkwardBackend())
    held = _read(inputs, first, session)
    assert _read(inputs, second, session).node_id == held.node_id


@pytest.mark.parametrize(
    "first, second",
    [
        ("events", "events;1"),
        ("events;1", "events|k"),
        ("events", "other"),
        ("events", "other|k"),
        ("<dir/events>", "<events>"),  # one name, another directory
        ("<events;1>", "events"),  # a lower cycle is not what a bare name opens
        ("<events;1>", "events|k"),
        # the same tree, but only opening the file could tell
        ("events|k", "events;2|k"),
        ("single|k", "single:events"),
        ("events:x|k", "events/x"),
    ],
)
@pytest.mark.parametrize("swap", [False, True])
def test_another_tree_or_an_undecided_spelling_is_refused(inputs, first, second, swap):
    if swap:
        first, second = second, first
    session = graphed.Session(AwkwardBackend())
    _read(inputs, first, session)
    with pytest.raises(ValueError, match="already bound to other data"):
        _read(inputs, second, session)
