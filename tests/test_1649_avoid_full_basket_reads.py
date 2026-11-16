# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""Regression tests for PR #1649.

- ``num_entries_for`` / memory-based ``step_size`` must not fully read and
  decompress baskets; only the ``TBasket`` ``TKeys`` should be read.
- A partial array-cache hit must not re-read (and re-decompress) the baskets of
  branches that were already cached.
"""

import pytest
import skhep_testdata

import uproot
import uproot.models.TBasket


def _count_basket_reads(monkeypatch):
    counter = {"n": 0}
    original = uproot.models.TBasket.Model_TBasket.read

    def spy(self, *args, **kwargs):
        counter["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(uproot.models.TBasket.Model_TBasket, "read", spy)
    return counter


def test_num_entries_for_does_not_fully_read_baskets(monkeypatch):
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as tree:
        num_baskets = sum(
            branch.num_baskets for branch in tree.itervalues(recursive=True)
        )
        assert num_baskets == 20
        counter = _count_basket_reads(monkeypatch)
        result = tree.num_entries_for("100 MB")

        # The previous implementation read (and decompressed) every basket here.
        assert counter["n"] == 0
        assert result == 696239


def test_partial_cache_hit_does_not_reread(monkeypatch):
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as tree:
        reference = tree.arrays(["px1", "px2"], array_cache={})

        cache = {}
        tree.arrays(["px1"], array_cache=cache)

        counter = _count_basket_reads(monkeypatch)
        out = tree.arrays(["px1", "px2"], array_cache=cache)

        # Only px2's basket should be read; px1 comes from the cache.
        assert counter["n"] == 1

        import awkward as ak

        assert ak.all(out["px1"] == reference["px1"])
        assert ak.all(out["px2"] == reference["px2"])


def test_full_cache_hit_reads_nothing(monkeypatch):
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as tree:
        cache = {}
        tree.arrays(["px1", "px2"], array_cache=cache)
        counter = _count_basket_reads(monkeypatch)
        out = tree.arrays(["px1", "px2"], array_cache=cache)
        assert counter["n"] == 0
        assert out["px1"][:2].tolist() == pytest.approx([-41.1952876442, 35.1180497674])
