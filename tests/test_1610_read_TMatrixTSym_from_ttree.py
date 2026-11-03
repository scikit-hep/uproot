# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy as np
import pytest
import skhep_testdata

import uproot


def test_read_TMatrixTSym_from_TTree():
    filepath = skhep_testdata.data_path("uproot-issue-1610.root")

    # branch contains `vector<TmatrixTSym<double>>`
    branch = uproot.open(filepath)["TMatrixTSymAnalysis/covMatrixOfTurn"]

    assert branch.file.streamer_named("TMatrixTSym<double>") is None

    arr_np = branch.array(library="np")
    arr_ak = branch.array(library="ak")

    assert len(arr_np) == len(arr_ak) == 8
    first = arr_ak[0][0]
    assert (first.fNrows, first.fNcols, first.fNelems) == (6, 6, 36)
    assert first.fElements[:3].tolist() == pytest.approx(
        [3.421351148343585, 0.12123125081178641, -0.14486934491502979]
    )

    for i in range(len(arr_np)):
        row_np = arr_np[i]
        row_ak = arr_ak[i]

        for j in range(len(row_np)):
            obj_np = row_np[j]
            obj_ak = row_ak[j]

            for field in obj_ak.fields:
                if field == "fElements":
                    assert np.all(obj_np.member(field) == obj_ak[field].to_numpy())
                else:
                    assert obj_np.member(field) == obj_ak[field]
