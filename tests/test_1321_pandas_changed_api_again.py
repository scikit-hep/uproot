# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata
import uproot

pytest.importorskip("pandas")
pytest.importorskip("awkward_pandas")


def test():
    df = uproot.concatenate(
        skhep_testdata.data_path("uproot-Zmumu.root"),
        library="pd",
        cut="Run > 148029",
    )
    assert df.shape == (1580, 20)
    assert df["Run"].unique().tolist() == [148031]
    assert df.index[:3].tolist() == [0, 1, 2]
    assert df["M"].iloc[0] == pytest.approx(82.4626915551)
