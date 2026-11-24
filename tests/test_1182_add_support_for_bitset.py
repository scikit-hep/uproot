# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import uproot
import skhep_testdata


def test_add_support_for_bitset():
    with uproot.open(skhep_testdata.data_path("uproot-issue-40.root")) as f:
        assert (
            repr(f["tree"]["bitset8"].interpretation)
            == "AsObjects(AsBitSet(True, dtype('bool')))"
        )
        assert (
            repr(f["tree"]["bitset16"].interpretation)
            == "AsObjects(AsBitSet(True, dtype('bool')))"
        )

        assert f["tree"]["bitset8"].array().tolist() == [
            [True, False, True, False, True, False, False, False]
        ]
        assert f["tree"]["bitset16"].array().tolist() == [
            [
                False,
                True,
                False,
                True,
                False,
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
            ]
        ]
