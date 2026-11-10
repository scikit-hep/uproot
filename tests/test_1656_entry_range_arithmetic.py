# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest

import uproot

awkward = pytest.importorskip("awkward")


def _write_multi_basket_tree(path, num_baskets=3, basket_size=10):
    with uproot.recreate(path) as f:
        f.mktree("t", {"x": "int64"})
        for i in range(num_baskets):
            f["t"].extend({"x": numpy.arange(i * basket_size, (i + 1) * basket_size)})


def test_basket_selection_no_extra_basket_on_boundary(tmp_path):
    """
    When ``entry_stop`` lands exactly on a basket boundary, the basket
    collector must not select a trailing basket that contributes zero
    entries.
    """
    path = str(tmp_path / "multi.root")
    _write_multi_basket_tree(path)

    with uproot.open(path) as f:
        branch = f["t"]["x"]
        assert branch.entry_offsets == [0, 10, 20, 30]

        # entry_stop on a boundary must not pull in the next basket
        assert [n for n, _ in branch.entries_to_ranges_or_baskets(0, 10)] == [0]
        assert [n for n, _ in branch.entries_to_ranges_or_baskets(0, 20)] == [0, 1]
        assert [n for n, _ in branch.entries_to_ranges_or_baskets(10, 20)] == [1]
        assert [n for n, _ in branch.entries_to_ranges_or_baskets(0, 30)] == [
            0,
            1,
            2,
        ]
        # non-boundary ranges still touch the right baskets
        assert [n for n, _ in branch.entries_to_ranges_or_baskets(5, 20)] == [0, 1]
        assert [n for n, _ in branch.entries_to_ranges_or_baskets(7, 23)] == [
            0,
            1,
            2,
        ]


def test_basket_boundary_reads_are_correct(tmp_path):
    """
    Results must be correct (and unchanged for non-boundary ranges) after the
    basket-selection fix.
    """
    path = str(tmp_path / "multi.root")
    _write_multi_basket_tree(path)

    with uproot.open(path) as f:
        branch = f["t"]["x"]
        for start, stop in [(0, 10), (0, 15), (5, 20), (10, 20), (0, 30), (7, 23)]:
            result = branch.array(entry_start=start, entry_stop=stop).tolist()
            assert result == list(range(start, stop))

        # empty ranges (including on boundaries) yield empty arrays
        for start, stop in [(0, 0), (10, 10), (15, 15), (30, 30)]:
            assert branch.array(entry_start=start, entry_stop=stop).tolist() == []


def test_basket_boundary_jagged_and_strings(tmp_path):
    """
    Jagged and string interpretations must also handle exact boundaries.
    """
    path = str(tmp_path / "jag.root")
    with uproot.recreate(path) as f:
        f.mktree("t", {"j": "var * int64", "s": "string"})
        for i in range(3):
            f["t"].extend(
                {
                    "j": awkward.Array(
                        [[i * 10 + k] * ((k % 3) + 1) for k in range(10)]
                    ),
                    "s": [f"str{i * 10 + k}" for k in range(10)],
                }
            )

    with uproot.open(path) as f:
        jagged = f["t"]["j"]
        strings = f["t"]["s"]
        for start, stop in [(0, 10), (5, 20), (10, 20), (0, 30), (7, 23)]:
            assert jagged.array(entry_start=start, entry_stop=stop).tolist() == [
                [k] * ((k % 10 % 3) + 1) for k in range(start, stop)
            ]
            result = strings.array(entry_start=start, entry_stop=stop).tolist()
            assert result == [f"str{k}" for k in range(start, stop)]


def test_concatenate_allow_missing_keeps_offsets(tmp_path):
    """
    ``concatenate(..., allow_missing=True)`` must keep the global entry
    offsets aligned when a file is skipped because of a missing branch.
    """
    files = []
    for i in range(3):
        path = str(tmp_path / f"file{i}.root")
        with uproot.recreate(path) as f:
            if i == 1:
                f.mktree("t", {"zzz": "int64"})
                f["t"].extend({"zzz": numpy.arange(10)})
            else:
                f.mktree("t", {"x": "int64"})
                f["t"].extend({"x": numpy.arange(i * 10, (i + 1) * 10)})
        files.append(path + ":t")

    arrays = uproot.concatenate(
        files,
        expressions=["x"],
        entry_start=5,
        entry_stop=25,
        allow_missing=True,
    )
    # file0[5:10] = 5..9 and file2[20:25] = 20..24; file1 is skipped
    assert arrays["x"].tolist() == [5, 6, 7, 8, 9, 20, 21, 22, 23, 24]


def test_rntuple_iterate_clamps_entry_stop(tmp_path):
    """
    ``RNTuple.iterate`` must clamp the per-step ``entry_stop`` to the
    user-requested ``entry_stop``.
    """
    path = str(tmp_path / "nt.root")
    with uproot.recreate(path) as f:
        f.mkrntuple("nt", {"x": numpy.dtype("int64")})
        f["nt"].extend({"x": numpy.arange(50000, dtype=numpy.int64)})

    with uproot.open(path)["nt"] as nt:
        total = 0
        ranges = []
        for arrays, report in nt.iterate(
            step_size=30000, entry_stop=40000, report=True
        ):
            total += len(arrays)
            ranges.append((report.start, report.stop))
        assert total == 40000
        assert ranges == [(0, 30000), (30000, 40000)]

        # entry_start together with entry_stop must also be respected
        chunks = list(nt.iterate(step_size=15000, entry_start=10000, entry_stop=40000))
        assert [len(arrays) for arrays in chunks] == [15000, 15000]
        assert chunks[0]["x"][0] == 10000
        assert chunks[1]["x"][-1] == 39999
