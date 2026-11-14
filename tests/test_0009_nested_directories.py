# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import sys

import numpy
import pytest
import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-nesteddirs.root")) as directory:
        assert directory.keys() == [
            "one;1",
            "one/two;1",
            "one/two/tree;1",
            "one/tree;1",
            "three;1",
            "three/tree;1",
        ]

        assert directory.path == ()
        assert directory["one"].path == ("one",)
        assert directory["one/two"].path == ("one", "two")
        assert directory["three;1"].path == ("three",)

        assert len(directory) == 6
        assert len(directory["one"]) == 3

        assert "one;1/two;1" in directory

        with pytest.raises(uproot.KeyInFileError, match="not found: 'whatever'"):
            directory["whatever"]

        assert directory.get("whatever") is None
