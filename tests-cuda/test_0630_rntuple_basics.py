# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
from __future__ import annotations

import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")
cupy = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    cupy.cuda.runtime.driverGetVersion() == 0, reason="No available CUDA driver."
)


@pytest.mark.parametrize(
    ("backend", "interpreter", "library"),
    [("cuda", "cpu", cupy), ("cuda", "gpu", cupy)],
)
def test_flat(backend, interpreter, library):
    filename = skhep_testdata.data_path("test_int_float_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        R = f["ntuple"]
        assert R.keys() == ["one_integers", "two_floats"]
        assert [r.type_name for r in R.header.field_records] == [
            "std::int32_t",
            "float",
        ]
        assert R.header.checksum == R.footer.header_checksum
        assert all(
            R.arrays(entry_stop=3, interpreter=interpreter, backend=backend)[
                "one_integers"
            ]
            == library.array([9, 8, 7])
        )
        assert all(
            R.arrays(
                "one_integers", entry_stop=3, interpreter=interpreter, backend=backend
            )["one_integers"]
            == library.array([9, 8, 7])
        )
        assert all(
            R.arrays(
                entry_start=1, entry_stop=3, interpreter=interpreter, backend=backend
            )["one_integers"]
            == library.array([8, 7])
        )

    filename = skhep_testdata.data_path("test_int_5e4_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        R = f["ntuple"]
        assert all(
            R.arrays(entry_stop=3, interpreter=interpreter, backend=backend)[
                "one_integers"
            ]
            == library.array([50000, 49999, 49998])
        )
        assert all(
            R.arrays(entry_start=-3, interpreter=interpreter, backend=backend)[
                "one_integers"
            ]
            == library.array([3, 2, 1])
        )
