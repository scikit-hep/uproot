# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Tests for RNTuple version 1.0.1.0 support (issue #1510).

RNTuple v1.0.1.0 added linked attribute sets to the footer structure.
This test ensures that both v1.0.1.0 files (with attributes) and v1.0.0.0
files (without attributes) can be read correctly.
"""

import pytest
import skhep_testdata
import xxhash

import uproot


@pytest.mark.parametrize(
    ("filename", "version_minor"),
    [
        ("ntpl001_staff_rntuple_v1-0-1-0.root", 1),
        ("ntpl001_staff_rntuple_v1-0-0-0.root", 0),
    ],
)
def test_rntuple_footer_parsing(filename, version_minor):
    """
    Test that RNTuple v1.0.1.0 files can be read, and that v1.0.0.0 files still can.

    The v1.0.1.0 format added a "linked attribute sets" list frame to the
    footer, appearing between cluster_group_records and the checksum.
    Without proper support, this causes an AssertionError due to checksum
    mismatch. In older files, the reader finds an empty list frame.
    """
    with uproot.open(skhep_testdata.data_path(filename)) as f:
        obj = f["Staff"]

        # Verify version
        assert obj.member("fVersionEpoch") == 1
        assert obj.member("fVersionMajor") == 0
        assert obj.member("fVersionMinor") == version_minor
        assert obj.member("fVersionPatch") == 0

        # Access footer - should not raise AssertionError
        footer = obj.footer

        # Neither file has attribute sets
        assert footer.linked_attribute_sets == []

        # Verify checksum validation passes
        chunk = obj._footer_chunk
        computed = xxhash.xxh3_64_intdigest(chunk.raw_data[:-8])
        stored = footer.checksum
        assert (
            computed == stored
        ), f"Footer checksum mismatch: {computed:016x} != {stored:016x}"

        # Verify we can retrieve keys (original failing operation)
        assert obj.keys() == [
            "Category",
            "Flag",
            "Age",
            "Service",
            "Children",
            "Grade",
            "Step",
            "Hrweek",
            "Cost",
            "Division",
            "Nation",
        ]

        # Verify normal operations still work
        assert obj.num_entries == 3354
        assert obj["Age"].array(entry_stop=5).tolist() == [58, 63, 56, 61, 52]
