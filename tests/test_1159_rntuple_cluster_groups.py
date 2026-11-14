# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test_multiple_cluster_groups():
    filename = skhep_testdata.data_path(
        "test_multiple_cluster_groups_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        assert len(obj.footer.cluster_group_records) == 3

        assert obj.footer.cluster_group_records[0].num_clusters == 5
        assert obj.footer.cluster_group_records[1].num_clusters == 4
        assert obj.footer.cluster_group_records[2].num_clusters == 3

        assert obj.num_entries == 1000

        arrays = obj.arrays()

        assert arrays.one.tolist() == list(range(1000))
        assert arrays.int_vector.tolist() == [[i, i + 1] for i in range(1000)]
