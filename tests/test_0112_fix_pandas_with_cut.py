# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

pandas = pytest.importorskip("pandas")


def test():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root")) as f:
        df = f["events"].arrays(
            ["px1", "pmag"],
            cut="pmag < 30",
            aliases={"pmag": "sqrt(px1**2 + py1**2)"},
            library="pd",
        )
        assert isinstance(df, pandas.DataFrame)  # not an empty tuple
        assert df.columns.tolist() == ["px1", "pmag"]
        assert len(df) == 551
        assert df.index[:3].tolist() == [4, 5, 6]
        assert df["px1"].iloc[0] == 22.7835819537
        assert df["pmag"].iloc[0] == 27.29809999997133
        assert (df["pmag"] < 30).all()
