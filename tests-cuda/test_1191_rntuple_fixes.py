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
def test_schema_extension(backend, interpreter, library):
    filename = skhep_testdata.data_path("test_extension_columns_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        assert len(obj.page_link_list[0]) < len(obj.page_link_list[1])

        assert len(obj.column_records) > len(obj.header.column_records)
        assert len(obj.column_records) == 4
        assert obj.column_records[1].first_element_index == 200
        assert obj.column_records[2].first_element_index == 400

        arrays = obj.arrays(backend=backend, interpreter=interpreter)

        assert len(arrays.float_field) == 600
        assert len(arrays.intvec_field) == 600

        assert ak.all(arrays.float_field[:200] == 0)
        assert ak.all(len(l) == 0 for l in arrays.intvec_field[:400])

        assert next(i for i, l in enumerate(arrays.float_field) if l != 0) == 200
        assert next(i for i, l in enumerate(arrays.intvec_field) if len(l) != 0) == 400


@pytest.mark.parametrize(
    ("backend", "interpreter", "library"),
    [("cuda", "cpu", cupy), ("cuda", "gpu", cupy)],
)
def test_rntuple_cardinality(backend, interpreter, library):
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        arrays = obj.arrays(backend=backend, interpreter=interpreter)
        assert ak.all(
            arrays["nMuon"] == library.array([len(l) for l in arrays["Muon_pt"]])
        )


@pytest.mark.parametrize(
    ("backend", "interpreter", "library"),
    [("cuda", "gpu", cupy)],
)
def test_multiple_page_delta_encoding_GDS(backend, interpreter, library):
    filename = skhep_testdata.data_path("test_index_multicluster_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]
        filehandle = uproot.source.cufile_interface.CuFileSource(filename, "rb")
        field_metadata = obj.get_field_metadata(0)
        col_clusterbuffers = obj.gpu_read_cluster_pages(
            0, 0, filehandle, field_metadata
        )
        filehandle.get_all()
        col_clusterbuffers._decompress()
        data = obj.gpu_deserialize_pages(col_clusterbuffers.data, 0, 0, field_metadata)
        assert data[64] - data[63] == 2


@pytest.mark.parametrize(
    ("backend", "interpreter", "library"),
    [("cuda", "cpu", cupy), ("cuda", "gpu", cupy)],
)
def test_split_encoding(backend, interpreter, library):
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        arrays = obj.arrays(backend=backend, interpreter=interpreter)

        expected_pt = library.array([10.763696670532227, 15.736522674560547])
        expected_charge = library.array([-1, -1])
        assert ak.all(arrays["Muon_pt"][0] == expected_pt)
        assert ak.all(arrays["Muon_charge"][0] == expected_charge)
