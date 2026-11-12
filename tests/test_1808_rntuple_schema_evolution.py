# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

import uproot


@pytest.mark.parametrize("automatic", [False, True], ids=["explicit", "automatic"])
def test_ntuple_add_fields_preserves_intervening_extends(tmp_path, automatic):
    path = tmp_path / "test.root"
    first = np.array([1, 2, 3], dtype=np.int32)
    second = np.array([4, 5], dtype=np.int32)

    with uproot.recreate(path) as f:
        ntuple = f.mkrntuple("n", {"x": np.int32})
        ntuple.add_fields({"y": np.int32})
        ntuple.extend({"x": first, "y": first + 10})

        if automatic:
            ntuple.extend(
                {"x": second, "y": second + 10, "z": second + 20},
                accept_new_fields=True,
            )
        else:
            ntuple.add_fields({"z": np.int32})

    with uproot.open(path) as f:
        ntuple = f["n"]
        if automatic:
            assert ntuple.num_entries == 5
            assert ak.to_list(ntuple["x"].array()) == [1, 2, 3, 4, 5]
            assert ak.to_list(ntuple["y"].array()) == [11, 12, 13, 14, 15]
            assert ak.to_list(ntuple["z"].array()) == [0, 0, 0, 24, 25]
        else:
            assert ntuple.num_entries == 3
            assert ak.to_list(ntuple["x"].array()) == [1, 2, 3]
            assert ak.to_list(ntuple["y"].array()) == [11, 12, 13]
            assert ak.to_list(ntuple["z"].array()) == [0, 0, 0]


def test_extend_invalidates_same_count_snapshot(tmp_path):
    path = tmp_path / "same-count.root"
    values = np.array([1, 2, 3], dtype=np.int32)

    with uproot.recreate(path) as f:
        ntuple = f.mkrntuple("n", {"x": np.int32})
        ntuple.extend({"x": values})
        ntuple.add_fields({"y": np.int32})
        ntuple._reload_cascading()
        cascade = ntuple._cascading
        # Model a stale snapshot whose count happens to match the next footer.
        stale_groups = cascade._existing_footer.cluster_group_records
        stale_groups.append(stale_groups[-1])
        ntuple.extend({"x": values + 3, "y": values + 10})
        assert len(stale_groups) == len(cascade._footer.cluster_group_record_frames)
        assert not hasattr(cascade, "_existing_footer")
        assert not hasattr(cascade, "_existing_page_list_envelopes")
        ntuple.add_fields({"z": np.int32})

    with uproot.open(path) as f:
        assert ak.to_list(f["n"]["x"].array()) == [1, 2, 3, 4, 5, 6]
        assert ak.to_list(f["n"]["y"].array()) == [0, 0, 0, 11, 12, 13]
        assert ak.to_list(f["n"]["z"].array()) == [0, 0, 0, 0, 0, 0]


def test_rejected_extend_keeps_existing_snapshot(tmp_path):
    path = tmp_path / "rejected.root"
    with uproot.recreate(path) as f:
        ntuple = f.mkrntuple("n", {"x": np.int32})
        ntuple.extend({"x": np.array([1, 2], dtype=np.int32)})
        ntuple.add_fields({"y": np.int32})
        ntuple._reload_cascading()
        cascade = ntuple._cascading
        footer = cascade._existing_footer
        page_lists = cascade._existing_page_list_envelopes
        with pytest.raises(ValueError):
            ntuple.extend({"x": np.array([3.5]), "y": np.array([1], dtype=np.int32)})
        assert cascade._existing_footer is footer
        assert cascade._existing_page_list_envelopes is page_lists
        ntuple.add_fields({"z": np.int32})

    with uproot.open(path) as f:
        assert ak.to_list(f["n"]["x"].array()) == [1, 2]
        assert ak.to_list(f["n"]["z"].array()) == [0, 0]
