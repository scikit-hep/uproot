# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata
import uproot

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("akimbo")


def test():
    assert (
        len(
            uproot.concatenate(
                skhep_testdata.data_path("uproot-Zmumu.root"),
                library="pd",
                cut="Run > 148029",
            )
        )
        == 1580
    )
