# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import uproot
import uproot.models.RNTuple as models
import uproot.writing._cascadentuple as cascade

ak = pytest.importorskip("awkward")


def _roundtrip(path, data):
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", data)
    with uproot.open(path) as file:
        return file["ntuple"].arrays()


def _read(reader, data):
    chunk = uproot.source.chunk.Chunk.wrap(None, np.frombuffer(data, dtype=np.uint8))
    return reader.read(chunk, uproot.source.cursor.Cursor(0), {})


def test_packed_union_does_not_change_the_input():
    layout = ak.Array([1, "a", 2, "b"])[:2].layout
    index = layout.index.data.copy()
    packed = cascade._to_packed(layout)
    assert [content.length for content in packed.contents] == [1, 1]
    assert layout.index.data.tolist() == index.tolist()


def test_packing_an_unknown_form():
    with pytest.raises(NotImplementedError, match="Form type .* cannot be written"):
        cascade._to_packed_form(object())


def test_packing_an_unknown_layout():
    with pytest.raises(NotImplementedError, match="Array type .* cannot be written"):
        cascade._to_packed(object())


def test_structured_array(tmp_path):
    data = np.array([(1, 2.5), (3, 4.5)], dtype=[("a", "i4"), ("b", "f8")])
    out = _roundtrip(tmp_path / "file.root", data)
    assert out.to_list() == [{"a": 1, "b": 2.5}, {"a": 3, "b": 4.5}]


def test_dataframe_with_object_columns(tmp_path):
    pandas = pytest.importorskip("pandas")
    data = pandas.DataFrame({"a": [[1, 2], [3]], "b": [1, 2]})
    out = _roundtrip(tmp_path / "file.root", data)
    assert out.to_list() == [
        {"index": 0, "a": [1, 2], "b": 1},
        {"index": 1, "a": [3], "b": 2},
    ]


@pytest.mark.parametrize("kind", ["series", "named index"])
def test_pandas_objects_that_are_not_tables(tmp_path, kind):
    pandas = pytest.importorskip("pandas")
    data = (
        pandas.Series([1, 2])
        if kind == "series"
        else pandas.DataFrame({"a": [1, 2]}, index=["x", "y"])
    )
    with uproot.recreate(tmp_path / "file.root") as file:
        with pytest.raises(TypeError, match="Input must be a type specification"):
            file.mkrntuple("ntuple", data)


def test_extend_with_something_that_is_not_data(tmp_path):
    with uproot.recreate(tmp_path / "file.root") as file:
        ntuple = file.mkrntuple("ntuple", {"x": np.array([1, 2])})
        with pytest.raises(TypeError, match="data must be an Awkward array"):
            ntuple.extend(5)


def test_list_frame_of_raw_items():
    payloads = [b"\x01" * 8, b"\x02" * 8]
    frame = cascade._serialize_rntuple_list_frame(payloads, wrap=True, rawinput=True)

    class Payload:
        def read(self, chunk, cursor, context):
            return cursor.bytes(chunk, 8, context).tobytes()

    reader = models.ListFrameReader(models.RecordFrameReader(Payload()))
    assert _read(reader, frame) == payloads


def test_record_frame_that_does_not_count_its_size():
    assert cascade._record_frame_wrap(b"abc", includeself=False) == (
        b"\x03\x00\x00\x00\x00\x00\x00\x00abc"
    )
    assert cascade._record_frame_wrap(b"abc") == (
        b"\x0b\x00\x00\x00\x00\x00\x00\x00abc"
    )


def test_default_field_names():
    form = ak.forms.RecordForm([ak.forms.NumpyForm("int32")], ["x"])
    header = cascade.NTuple_Header(None, "ntuple", "", form)
    header._build_field_col_records(ak.forms.NumpyForm("float32"))
    assert header._field_records[-1].field_name == "_1"
    assert header._field_records[-1].type_name == "float"
    assert header._field_records[-1].parent_field_id == 1


def test_cluster_group_record():
    locator = cascade.NTuple_Locator(16, 100)
    envlink = cascade.NTuple_EnvLink(48, locator)
    record = cascade.NTuple_ClusterGroupRecord(2, 30, 4, envlink)
    assert record.min_entry == 2
    assert record.entry_span == 30
    assert record.num_clusters == 4
    assert record.page_list_envlink is envlink
    assert (
        repr(record)
        == "NTuple_ClusterGroupRecord(4, NTuple_EnvLink(48, NTuple_Locator(16, 100)))"
    )

    out = _read(models.ClusterGroupRecordReader(), record.serialize())
    assert out.min_entry_num == 2
    assert out.entry_span == 30
    assert out.num_clusters == 4
    assert out.page_list_link.env_uncomp_size == 48
    assert out.page_list_link.locator.num_bytes == 16
    assert out.page_list_link.locator.offset == 100


def test_page_reprs():
    locator = cascade.NTuple_Locator(16, 100)
    assert repr(locator) == "NTuple_Locator(16, 100)"
    assert repr(cascade.NTuple_ClusterSummary(1, 2)) == "NTuple_ClusterSummary(1, 2, 0)"

    page = cascade.NTuple_PageDescription(3, locator)
    assert repr(page) == "NTuple_PageDescription(3, NTuple_Locator(16, 100))"
    assert repr(cascade.NTuple_ColumnPageListDescription([page], 5, 505)) == (
        "NTuple_ColumnPageListDescription("
        "[NTuple_PageDescription(3, NTuple_Locator(16, 100))], 5, 505)"
    )


def test_cascading_ntuple_properties(tmp_path):
    with uproot.recreate(tmp_path / "file.root") as file:
        writable = file.mkrntuple("ntuple", {"x": np.array([1, 2])}, description="doc")
        cascading = writable._cascading
        assert cascading.directory is file._cascading
        assert cascading.name.string == "ntuple"
        # the description goes in the header, not in the TKey
        assert cascading.title.string == "ntuple"
        assert cascading.freesegments is file._cascading._freesegments
        assert cascading.location == cascading.key.location
        assert cascading.num_entries == 2
