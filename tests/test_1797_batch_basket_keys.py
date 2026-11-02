# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
A memory-based ``step_size`` must read all ``TBasket`` ``TKeys`` in one
request, not one round trip per basket.
"""

from __future__ import annotations

import skhep_testdata

import uproot


def _total_uncompressed_bytes(path, tree_name):
    # A separate open, so the key size cache of the tree under test stays empty
    with uproot.open(path)[tree_name] as tree:
        return sum(
            branch.basket_uncompressed_bytes(i)
            for branch in tree.itervalues(recursive=True)
            for i in range(branch.num_baskets)
        )


def test_num_entries_for_reads_keys_in_one_request(monkeypatch):
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    expected_bytes = _total_uncompressed_bytes(path, "events")
    with uproot.open(path)["events"] as tree:
        num_baskets = sum(
            branch.num_baskets for branch in tree.itervalues(recursive=True)
        )
        assert num_baskets > 1

        source = tree.file.source
        calls = {"chunk": 0, "chunks": 0}
        original_chunk = source.chunk
        original_chunks = source.chunks

        def chunk(start, stop):
            calls["chunk"] += 1
            return original_chunk(start, stop)

        def chunks(ranges, notifications, **kwargs):
            calls["chunks"] += 1
            assert all(type(x) is int for r in ranges for x in r)
            return original_chunks(ranges, notifications, **kwargs)

        monkeypatch.setattr(source, "chunk", chunk)
        monkeypatch.setattr(source, "chunks", chunks)

        result = tree.num_entries_for(expected_bytes)

        assert calls == {"chunk": 0, "chunks": 1}
        assert result == tree.num_entries


def test_num_entries_for_fsspec_does_not_coalesce_keys(monkeypatch):
    path = skhep_testdata.data_path("uproot-HZZ.root")
    expected_bytes = _total_uncompressed_bytes(path, "events")
    with uproot.open(f"file://{path}", handler=uproot.source.fsspec.FSSpecSource)[
        "events"
    ] as tree:
        branches = list(tree.itervalues(recursive=True))
        num_baskets = sum(branch.num_baskets for branch in branches)
        assert num_baskets > 1

        source = tree.file.source
        fetched = []
        # fsspec caches filesystem instances, so patch the class through
        # monkeypatch to keep the spy from leaking into other tests.
        fs_class = type(source._fs)
        original_cat_ranges = fs_class.cat_ranges

        def cat_ranges(self, paths, starts, ends, **kwargs):
            fetched.extend(stop - start for start, stop in zip(starts, ends))
            return original_cat_ranges(self, paths, starts, ends, **kwargs)

        monkeypatch.setattr(fs_class, "cat_ranges", cat_ranges)
        monkeypatch.setattr(source, "_async_impl", False)

        assert tree.num_entries_for(expected_bytes) == tree.num_entries
        assert sum(fetched) == num_baskets * uproot.reading._key_format_big.size


def test_num_entries_for_entry_range():
    path = skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    with uproot.open(path)["sample"] as tree:
        branch = tree["n"]
        assert branch.num_baskets > 2
        start, stop = branch.entry_offsets[1], branch.entry_offsets[2]
        expected_bytes = sum(
            b.basket_uncompressed_bytes(i)
            for b in tree.itervalues(recursive=True)
            for i in range(b.num_baskets)
            if b.entry_offsets[i] < stop and start < b.entry_offsets[i + 1]
        )
        assert (
            tree.num_entries_for(expected_bytes, entry_start=start, entry_stop=stop)
            == stop - start
        )


def _spy_reads(monkeypatch, source):
    calls = {"chunk": 0, "chunks": 0}
    original_chunk = source.chunk
    original_chunks = source.chunks

    def chunk(start, stop):
        calls["chunk"] += 1
        return original_chunk(start, stop)

    def chunks(ranges, notifications, **kwargs):
        calls["chunks"] += 1
        return original_chunks(ranges, notifications, **kwargs)

    monkeypatch.setattr(source, "chunk", chunk)
    monkeypatch.setattr(source, "chunks", chunks)
    return calls


def test_num_entries_for_caches_key_sizes(monkeypatch):
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as tree:
        calls = _spy_reads(monkeypatch, tree.file.source)
        first = tree.num_entries_for("1 kB")
        assert calls == {"chunk": 0, "chunks": 1}

        assert tree.num_entries_for("1 kB") == first
        branch = next(tree.itervalues())
        branch.basket_uncompressed_bytes(0)
        assert calls == {"chunk": 0, "chunks": 1}


def test_basket_uncompressed_bytes_fills_cache(monkeypatch):
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as tree:
        expected = tree.num_entries_for("1 kB")

    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as tree:
        for branch in tree.itervalues(recursive=True):
            for i in range(branch.num_baskets):
                branch.basket_uncompressed_bytes(i)

        calls = _spy_reads(monkeypatch, tree.file.source)
        assert tree.num_entries_for("1 kB") == expected
        assert calls == {"chunk": 0, "chunks": 0}
