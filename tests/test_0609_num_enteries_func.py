# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata
import uproot


def test_num_entries_single():
    file1 = skhep_testdata.data_path("uproot-Zmumu.root")
    ttree1 = file1 + ":events"
    assert list(uproot.num_entries(ttree1)) == [(file1, "events", 2304)]


@pytest.mark.parametrize(
    "files",
    [
        lambda a, b, c: [f"{a}:events", f"{b}:events", f"{c}:sample"],
        lambda a, b, c: {a: "events", b: "events", c: "sample"},
    ],
    ids=["list", "dict"],
)
def test_num_entries_multiple(files):
    file1 = skhep_testdata.data_path("uproot-Zmumu.root")
    file2 = skhep_testdata.data_path("uproot-HZZ.root")
    file3 = skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root")

    result = uproot.num_entries(files(file1, file2, file3))

    # entries are produced one at a time, not collected in a list first
    assert iter(result) is result
    assert list(result) == [
        (file1, "events", 2304),
        (file2, "events", 2421),
        (file3, "sample", 30),
    ]
