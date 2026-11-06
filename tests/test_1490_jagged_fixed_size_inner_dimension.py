# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import uproot

ak = pytest.importorskip("awkward")


def _batches(inner_shape, dtype):
    """Three batches of ``var * inner_shape * dtype``, with empty lists and an all-empty batch."""
    rng = np.random.default_rng(1490)
    return [
        ak.unflatten((rng.random((sum(c), *inner_shape)) * 100).astype(dtype), c)
        for c in ([0, 3, 1, 0, 2], [0, 0, 0], [5, 0, 1, 4])
    ]


def _write(tmp_path, branches):
    """Write ``{name: batches}`` with one ``extend`` (so one basket) per batch."""
    path = str(tmp_path / "test.root")
    with uproot.recreate(path) as f:
        tree = f.mktree("tree", {k: v[0].type.content for k, v in branches.items()})
        for batch in zip(*branches.values()):
            tree.extend(dict(zip(branches, batch)))
    return path


def _check(path, branches):
    with uproot.open(path) as f:
        for name, batches in branches.items():
            assert f["tree"][name].num_baskets == len(batches)
            assert f["tree"][name].array().tolist() == ak.concatenate(batches).tolist()


@pytest.mark.parametrize(
    ("inner_shape", "dtype"),
    [((1,), np.float32), ((6,), np.float64), ((2, 3), np.int32)],
)
def test_round_trip(tmp_path, inner_shape, dtype):
    branches = {"x": _batches(inner_shape, dtype)}
    _check(_write(tmp_path, branches), branches)


def test_unusual_layouts(tmp_path):
    flat = np.arange(60.0).reshape(10, 6)
    numpy_inner = ak.unflatten(ak.from_numpy(flat, regulararray=False), [3, 0, 2, 5])
    regular = ak.unflatten(ak.from_numpy(flat, regulararray=True), [3, 0, 2, 5])
    # a RegularArray whose content is longer than size * length
    overlong = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 3, 5, 10])),
        ak.contents.RegularArray(ak.contents.NumpyArray(np.arange(64.0)), 6),
    )
    # inner dimension as a NumPy inner shape, a ListArray, offsets not starting at 0
    branches = {
        "x": [numpy_inner, numpy_inner[[3, 0, 2]], regular[2:], ak.Array(overlong)]
    }
    _check(_write(tmp_path, branches), branches)


def test_update_and_extend(tmp_path):
    branches = {"a": _batches((6,), np.float64), "b": _batches((2, 3), np.int32)}
    path = _write(tmp_path, {k: v[:1] for k, v in branches.items()})
    for i in (1, 2):
        with uproot.update(path) as f:
            f["tree"].extend({k: v[i] for k, v in branches.items()})
    _check(path, branches)


def test_read_with_ROOT(tmp_path, capfd):
    ROOT = pytest.importorskip("ROOT")
    branches = {
        "d6": _batches((6,), np.float64),
        "f3": _batches((3,), np.float32),
        "i23": _batches((2, 3), np.int32),
    }
    file = ROOT.TFile.Open(_write(tmp_path, branches))
    tree = file.Get("tree")
    assert tree.GetEntries() == 12
    for name, batches in branches.items():
        expected = ak.concatenate(batches).tolist()
        leaf = tree.GetLeaf(name)
        for i in range(tree.GetEntries()):
            tree.GetEntry(i)
            values = [leaf.GetValue(j) for j in range(leaf.GetLen())]
            assert values == pytest.approx(np.ravel(expected[i]).tolist())
    file.Close()
    assert "repair" not in "".join(capfd.readouterr())
