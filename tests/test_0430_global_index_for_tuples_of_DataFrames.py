# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("pandas")
pytest.importorskip("awkward_pandas")


def test():
    frames = list(
        uproot.iterate(
            [skhep_testdata.data_path("uproot-HZZ-uncompressed.root") + ":events"] * 2,
            ["Muon_Px", "Jet_Px", "MET_px"],
            library="pd",
        )
    )

    # the index keeps counting from one file to the next
    assert [(df.index[0], df.index[-1]) for df in frames] == [(0, 2420), (2421, 4841)]
    for df in frames:
        assert df.columns.tolist() == ["Muon_Px", "Jet_Px", "MET_px"]
        assert len(df) == 2421
        assert df["Muon_Px"].iloc[0].tolist() == pytest.approx(
            [-52.89945602416992, 37.7377815246582]
        )
        assert df["Jet_Px"].iloc[1].tolist() == pytest.approx([-38.87471389770508])
        assert df["MET_px"].iloc[1] == pytest.approx(24.765203)
