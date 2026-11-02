# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import pytest
import skhep_testdata

import uproot
from uproot.source.cufile_interface import CuFileSource

ak = pytest.importorskip("awkward")
cupy = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    cupy.cuda.runtime.driverGetVersion() == 0, reason="No available CUDA driver."
)

UNCOMPRESSED = "rntviewer-testfile-uncomp-single-rntuple-v1-0-0-0.root"


def test_optional_fields(tmp_path):
    path = tmp_path / "optional.root"
    # missing entries first, in a row, and last
    data = ak.Array(
        {
            "x": [None, 2, None, None, 5, 6, None],
            "y": [[1.5], None, [], [2.5, 3.5], None, [4.5], [5.5]],
        }
    )
    # the GPU interpreter can decompress ZSTD and LZ4, but not ZLIB
    with uproot.recreate(path, compression=uproot.ZSTD(1)) as file:
        file.mkrntuple("ntuple", data)

    with uproot.open(path) as file:
        arrays = file["ntuple"].arrays(backend="cuda", interpreter="gpu")
    assert ak.backend(arrays) == "cuda"
    assert ak.to_backend(arrays, "cpu").to_list() == data.to_list()


def test_uncompressed_pages_and_cufile_counters():
    filename = skhep_testdata.data_path(UNCOMPRESSED)
    with uproot.open(filename) as file:
        ntuple = file["Contributors"]
        # column 1 holds the characters of firstName in one page of 178 bytes
        field_metadata = ntuple.get_field_metadata(1)
        filehandle = CuFileSource(filename, "rb")
        assert filehandle.file_path == filename
        payload = ntuple.gpu_read_cluster_pages(0, 1, filehandle, field_metadata)
        assert filehandle.num_requested_chunks == 1
        assert filehandle.num_requested_bytes == 178
        assert filehandle.requested_chunk_sizes == [178]
        assert len(filehandle.futures) == 1
        filehandle.get_all()
        filehandle.close()

        assert not payload.page_is_compressed
        payload._decompress()  # nothing to decompress, the pages were read as is
        data = ntuple.gpu_deserialize_pages(payload.data, 1, 0, field_metadata)
        expected = ntuple.read_cluster_range(1, 0, 1)
        assert cupy.asnumpy(data).tobytes() == expected.tobytes()
        assert bytes(expected[:5]) == b"Jakob"


def test_uncompressed_arrays():
    with uproot.open(skhep_testdata.data_path(UNCOMPRESSED)) as file:
        ntuple = file["Contributors"]
        arrays = ntuple.arrays(backend="cuda", interpreter="gpu")
        expected = ntuple.arrays()
    assert len(expected) == 22
    assert ak.to_backend(arrays, "cpu").to_list() == expected.to_list()
