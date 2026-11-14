# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
from __future__ import annotations

import numpy
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


def test_direct_reads_from_before_the_first_cluster(tmp_path):
    path = tmp_path / "flat.root"
    with uproot.recreate(path, compression=uproot.ZSTD(1)) as file:
        file.mkrntuple("ntuple", {"x": numpy.array([1, 2, 3], numpy.int64)})

    with uproot.open(path) as file:
        ntuple = file["ntuple"]

        def read(start, stop):
            # without cluster_offset, which is then taken from the clusters
            datas = ntuple.gpu_read_cluster_range(["column-0"], start, stop)
            datas._decompress()
            arrays = ntuple.gpu_deserialize_decompressed_content(datas, start, stop)
            return {key: cupy.asnumpy(value).tolist() for key, value in arrays.items()}

        # index -1 is before the first cluster, so it adds nothing
        assert read(0, 1) == {0: [1, 2, 3]}
        assert read(-1, 1) == {0: [1, 2, 3]}
        assert read(-1, 0) == {}


def test_narrower_representation_is_converted_by_value():
    # column 1 is a float16 representation of a field whose result is float32
    filename = skhep_testdata.data_path(
        "test_multiple_representations_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as file:
        ntuple = file["ntuple"]
        assert ntuple.get_field_metadata(1).dtype == numpy.dtype("float16")
        assert ntuple.get_field_metadata(1).dtype_result == numpy.dtype("float32")
        datas = ntuple.gpu_read_cluster_range(["column-1"], 0, 3)
        datas._decompress()
        arrays = ntuple.gpu_deserialize_decompressed_content(datas, 0, 3)
        expected = ntuple.read_cluster_range(1, 0, 3)
    assert cupy.asnumpy(arrays[1]).tolist() == expected.tolist() == [1.0, 2.0, 3.0]


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

        assert payload.pages == [None]
        payload._decompress()  # nothing to decompress, the pages were read as is
        data = ntuple.gpu_deserialize_pages(payload.data, 1, 0, field_metadata)
        expected = ntuple.read_cluster_range(1, 0, 1)
        assert cupy.asnumpy(data).tobytes() == expected.tobytes()
        assert bytes(expected[:5]) == b"Jakob"


@pytest.mark.parametrize(
    "compression_settings",
    [0, 5, 100 * uproot.const.kZSTD],  # nothing, algorithm 0, level 0
)
def test_compression_settings_of_uncompressed_pages(compression_settings):
    filename = skhep_testdata.data_path(UNCOMPRESSED)
    with uproot.open(filename) as file:
        ntuple = file["Contributors"]
        ntuple.page_link_list[0][1].compression_settings = compression_settings
        field_metadata = ntuple.get_field_metadata(1)
        filehandle = CuFileSource(filename, "rb")
        payload = ntuple.gpu_read_cluster_pages(0, 1, filehandle, field_metadata)
        filehandle.get_all()
        filehandle.close()

        assert payload.algorithm is None
        assert payload.compression_level == compression_settings % 100
        payload._decompress()
        data = ntuple.gpu_deserialize_pages(payload.data, 1, 0, field_metadata)
        expected = ntuple.read_cluster_range(1, 0, 1)
        assert cupy.asnumpy(data).tobytes() == expected.tobytes()


def test_uncompressed_arrays():
    with uproot.open(skhep_testdata.data_path(UNCOMPRESSED)) as file:
        ntuple = file["Contributors"]
        arrays = ntuple.arrays(backend="cuda", interpreter="gpu")
        expected = ntuple.arrays()
    assert len(expected) == 22
    assert ak.to_backend(arrays, "cpu").to_list() == expected.to_list()


@pytest.mark.parametrize(
    ("compression", "first", "compressed"),
    [
        # every page is read as is
        (None, [True, False, True] * 333 + [True], [False, False]),
        # the first page is compressed, but the second is too small for it
        (uproot.ZSTD(1), [True] * 1000, [True, False]),
        # the column is compressed, but neither of its pages
        (uproot.ZSTD(1), [True, False, False] * 4 + [True], [False, False]),
    ],
)
def test_bit_column_of_many_pages(tmp_path, compression, first, compressed):
    # uproot writes one page per cluster, so the pages of two clusters are made
    # the pages of the first; their numbers of bits are not multiples of 8
    second = [True, False, True, True, False]
    path = tmp_path / "bits.root"
    with uproot.recreate(path, compression=compression) as file:
        ntuple = file.mkrntuple("ntuple", {"b": numpy.array(first)})
        ntuple.extend({"b": numpy.array(second)})

    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        field_metadata = ntuple.get_field_metadata(0)
        assert field_metadata.isbit
        first_pages = ntuple.page_link_list[0][0].pages
        first_pages.extend(ntuple.page_link_list[1][0].pages)
        assert [page.num_elements for page in first_pages] == [len(first), 5]
        expected = first + second
        assert ntuple.read_cluster_pages(0, 0, field_metadata).tolist() == expected

        datas = ntuple.gpu_read_cluster_range(["column-0"], 0, 1)
        payload = datas.refs[0].fieldpayloads[0]
        assert [page is not None for page in payload.pages] == compressed
        datas._decompress()
        arrays = ntuple.gpu_deserialize_decompressed_content(datas, 0, 1)
    assert cupy.asnumpy(arrays[0]).tolist() == expected


@pytest.mark.parametrize("compression", [None, uproot.ZSTD(1)])
def test_bit_column_payload_decompresses_itself(tmp_path, compression):
    path = tmp_path / "bits.root"
    values = [True] * 1000 + [False] * 3
    with uproot.recreate(path, compression=compression) as file:
        file.mkrntuple("ntuple", {"b": numpy.array(values)})

    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        field_metadata = ntuple.get_field_metadata(0)
        filehandle = CuFileSource(str(path), "rb")
        payload = ntuple.gpu_read_cluster_pages(0, 0, filehandle, field_metadata)
        filehandle.get_all()
        filehandle.close()
        assert any(page is not None for page in payload.pages) == (
            compression is not None
        )
        payload._decompress()
        data = ntuple.gpu_deserialize_pages(payload.data, 0, 0, field_metadata)
    assert cupy.asnumpy(data).tolist() == values
