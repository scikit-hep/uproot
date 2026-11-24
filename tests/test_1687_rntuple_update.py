# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os
import shutil

import awkward as ak
import numpy as np
import pytest
import skhep_testdata
import uproot

try:
    import ROOT

    has_root = True
except ImportError:
    has_root = False


def test_extend_existing_ntuple(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {
            "x": np.array([1, 2, 3, 4, 5], dtype=np.float32),
            "y": np.array([10, 20, 30, 40, 50], dtype=np.int32),
        }

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend(
            {
                "x": np.array([6, 7, 8], dtype=np.float32),
                "y": np.array([60, 70, 80], dtype=np.int32),
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(
            nt["x"].array() == np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        )
        assert ak.all(
            nt["y"].array()
            == np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
        )

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 8
        entry = reader.CreateEntry()
        vals = []
        for i in range(reader.GetNEntries()):
            reader.LoadEntry(i, entry)
            vals.append(entry["x"])
        assert vals == pytest.approx([1, 2, 3, 4, 5, 6, 7, 8])


def test_add_field_ntuple(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3, 4, 5], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"z": np.int32})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(nt["x"].array() == np.array([1, 2, 3, 4, 5], dtype=np.float32))
        assert ak.all(nt["z"].array() == np.zeros(5, dtype=np.int32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 5
        entry = reader.CreateEntry()
        vals = []
        for i in range(reader.GetNEntries()):
            reader.LoadEntry(i, entry)
            vals.append(entry["x"])
        assert vals == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_add_field_ntuple_same_session_as_creation(tmp_path):
    """add_fields() on an RNTuple created earlier in the same session (not reopened).

    Regression test: the NTuple cascading object only gets its
    _existing_footer/_existing_page_list_envelopes/_existing_field_records
    attributes set inside _load_existing_ntuple, the path used when
    uproot.update() reconstructs a preexisting RNTuple from disk. An RNTuple
    created via mkrntuple or directory assignment earlier in the same
    session never goes through that path, so add_fields() raised a raw
    AttributeError reading self._cascading._existing_footer.
    """
    path = os.path.join(tmp_path, "test.root")
    with uproot.recreate(path) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3, 4, 5], dtype=np.float32)}
        f["mytuple"].add_fields({"z": np.int32})

    with uproot.open(path) as f:
        nt = f["mytuple"]
        assert ak.all(nt["x"].array() == np.array([1, 2, 3, 4, 5], dtype=np.float32))
        assert ak.all(nt["z"].array() == np.zeros(5, dtype=np.int32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", path)
        assert reader.GetNEntries() == 5


def test_add_field_ntuple_duplicate(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError, match="already exists"):
            f["mytuple"].add_fields({"x": np.int32})


def test_ntuple_extend_docstring_present():
    """WritableNTuple.extend must have a docstring.

    Regression test: the _column_encoding_error check was briefly inserted as
    the first statement in extend()'s body, before its docstring, which
    turned the docstring into a dangling (unused) string expression instead
    of __doc__ -- silently dropping the method from generated docs.
    """
    doc = uproot.writing.writable.WritableNTuple.extend.__doc__
    assert doc.lstrip().startswith("Args:")


@pytest.mark.parametrize(
    "filename",
    ["ntpl001_staff_rntuple_v1-0-1-0.root", "ntpl001_staff_rntuple_v1-0-0-0.root"],
)
def test_add_fields_rejects_root_written_column_encoding(tmp_path, filename):
    """add_fields()/extend() on a genuinely ROOT-written RNTuple must raise cleanly.

    ROOT commonly uses column encodings (e.g. split encoding) that Uproot does
    not write, so mutating such a file would either produce an inconsistent
    RNTuple or require Uproot to guess at an encoding it can't reproduce.
    Confirms the encoding guard actually protects a real ROOT-written file
    (not just a synthetic mismatch), matching the documented restriction that
    extend()/add_fields() on a reopened file only work reliably for RNTuples
    Uproot itself wrote.
    """
    path = os.path.join(tmp_path, "staff.root")
    shutil.copy(skhep_testdata.data_path(filename), path)

    with uproot.update(path) as f:
        with pytest.raises(ValueError, match="column encodings"):
            f["Staff"].add_fields({"newfield": np.int32})


def test_extend_ntuple_multiple_times(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend({"x": np.array([4, 5, 6], dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend({"x": np.array([7, 8, 9], dtype=np.float32)})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert ak.all(
            f["mytuple"]["x"].array()
            == np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
        )

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 9


def test_add_multiple_fields_ntuple(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"y": np.int32, "z": np.float64})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert ak.all(
            f["mytuple"]["x"].array() == np.array([1, 2, 3], dtype=np.float32)
        )
        assert ak.all(f["mytuple"]["y"].array() == np.zeros(3, dtype=np.int32))
        assert ak.all(f["mytuple"]["z"].array() == np.zeros(3, dtype=np.float64))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 3


def test_extend_ntuple_wrong_fields(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {
            "x": np.array([1, 2, 3], dtype=np.float32),
            "y": np.array([4, 5, 6], dtype=np.int32),
        }

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError, match="Data is not compatible"):
            f["mytuple"].extend(
                {"x": np.array([7, 8, 9], dtype=np.float32)}
            )  # missing y


def test_ntuple_dtypes(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {
            "x_f32": np.array([1, 2, 3], dtype=np.float32),
            "x_f64": np.array([1, 2, 3], dtype=np.float64),
            "x_i32": np.array([1, 2, 3], dtype=np.int32),
            "x_i64": np.array([1, 2, 3], dtype=np.int64),
        }

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields(
            {
                "z_f32": np.float32,
                "z_f64": np.float64,
                "z_i32": np.int32,
                "z_i64": np.int64,
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        arrays = f["mytuple"].arrays()
        assert {field: str(arrays[field].type) for field in arrays.fields} == {
            "x_f32": "3 * float32",
            "x_f64": "3 * float64",
            "x_i32": "3 * int32",
            "x_i64": "3 * int64",
            "z_f32": "3 * float32",
            "z_f64": "3 * float64",
            "z_i32": "3 * int32",
            "z_i64": "3 * int64",
        }
        for field in arrays.fields:
            expected = [1, 2, 3] if field.startswith("x") else [0, 0, 0]
            assert arrays[field].tolist() == expected

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 3


def test_ntuple_variable_length(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {
            "x": ak.Array([[1, 2], [3, 4, 5], [6]]),
        }

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend({"x": ak.Array([[7, 8, 9], [10]])})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert ak.all(
            f["mytuple"]["x"].array()
            == ak.Array([[1, 2], [3, 4, 5], [6], [7, 8, 9], [10]])
        )

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 5


def test_ntuple_mixed_types_extend(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {
            "pt": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            "jets": ak.Array([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]),
        }

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend(
            {
                "pt": np.array([40.0, 50.0], dtype=np.float32),
                "jets": ak.Array([[7.0, 8.0, 9.0], [10.0]]),
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(
            nt["pt"].array() == np.array([10, 20, 30, 40, 50], dtype=np.float32)
        )
        assert ak.all(
            nt["jets"].array() == ak.Array([[1, 2], [3], [4, 5, 6], [7, 8, 9], [10]])
        )

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 5
        entry = reader.CreateEntry()
        vals = []
        for i in range(reader.GetNEntries()):
            reader.LoadEntry(i, entry)
            vals.append(entry["pt"])
        assert vals == pytest.approx([10.0, 20.0, 30.0, 40.0, 50.0])


def test_ntuple_extend_empty(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mkrntuple("mytuple", {"x": np.dtype("float32")})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend({"x": np.array([1, 2, 3], dtype=np.float32)})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert nt.num_entries == 3
        assert ak.all(nt["x"].array() == np.array([1, 2, 3], dtype=np.float32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 3


def test_ntuple_multiple_in_file(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["tuple1"] = {"x": np.array([1, 2, 3], dtype=np.float32)}
        f["tuple2"] = {"y": np.array([4, 5, 6], dtype=np.int32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tuple1"].extend({"x": np.array([4, 5], dtype=np.float32)})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert ak.all(
            f["tuple1"]["x"].array() == np.array([1, 2, 3, 4, 5], dtype=np.float32)
        )
        assert ak.all(f["tuple2"]["y"].array() == np.array([4, 5, 6], dtype=np.int32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader1 = ROOT.RNTupleReader.Open("tuple1", os.path.join(tmp_path, "test.root"))
        reader2 = ROOT.RNTupleReader.Open("tuple2", os.path.join(tmp_path, "test.root"))
        assert reader1.GetNEntries() == 5
        assert reader2.GetNEntries() == 3


def test_ntuple_multiple_add_fields_then_extend(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"y": np.int32})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"z": np.float64})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend(
            {
                "x": np.array([4, 5, 6], dtype=np.float32),
                "y": np.array([40, 50, 60], dtype=np.int32),
                "z": np.array([400.0, 500.0, 600.0], dtype=np.float64),
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(nt["x"].array() == np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
        assert ak.all(
            nt["y"].array() == np.array([0, 0, 0, 40, 50, 60], dtype=np.int32)
        )
        assert ak.all(
            nt["z"].array() == np.array([0, 0, 0, 400, 500, 600], dtype=np.float64)
        )

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 6


def test_ntuple_add_field_and_extend_same_session(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"y": np.int32})
        f["mytuple"].extend(
            {
                "x": np.array([4, 5], dtype=np.float32),
                "y": np.array([40, 50], dtype=np.int32),
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(nt["x"].array() == np.array([1, 2, 3, 4, 5], dtype=np.float32))
        assert ak.all(nt["y"].array() == np.array([0, 0, 0, 40, 50], dtype=np.int32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 5
        entry = reader.CreateEntry()
        vals = []
        for i in range(reader.GetNEntries()):
            reader.LoadEntry(i, entry)
            vals.append(entry["x"])
        assert vals == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_ntuple_add_field_and_extend_in_subdirectory(tmp_path):
    """add_fields()/extend() on an RNTuple that lives in a subdirectory, not the root.

    Regression test: both methods resolved their own key by looking it up
    directly in the file's root directory using only the last path component
    (self._path[-1]), ignoring any subdirectory prefix. For an RNTuple in a
    subdirectory, that lookup found no such key at the root and returned
    None, and the next attribute access on it (key.seek_location) raised
    AttributeError -- even though the write that triggered the reload had
    already completed successfully, making it a spurious exception.
    """
    path = os.path.join(tmp_path, "test.root")
    with uproot.recreate(path) as f:
        f.mkdir("sub")
        f["sub/mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(path) as f:
        f["sub/mytuple"].add_fields({"y": np.int32})
        f["sub/mytuple"].extend(
            {
                "x": np.array([4, 5], dtype=np.float32),
                "y": np.array([40, 50], dtype=np.int32),
            }
        )

    with uproot.open(path) as f:
        nt = f["sub/mytuple"]
        assert ak.all(nt["x"].array() == np.array([1, 2, 3, 4, 5], dtype=np.float32))
        assert ak.all(nt["y"].array() == np.array([0, 0, 0, 40, 50], dtype=np.int32))


def test_ntuple_multiple_add_fields_same_session(tmp_path):
    """Repeated add_fields() calls within a single uproot.update() session.

    Regression test: the in-memory field-record refresh at the end of
    add_fields() concatenated the *already-updated* existing_field_records
    (which, from the second call on, already includes every field added by
    a previous call in this session) with footer.extension_field_record_frames
    (which is itself cumulative across the whole session, since the footer is
    rewritten from scratch each time). That double-counted every previously
    added field: 'a' would appear twice after the second call, next_field_id
    would be inflated, and the ids assigned to a third call's new field would
    exceed the number of fields actually on disk, corrupting the file (a
    third add_fields raised IndexError on reopen, at
    uproot.models.RNTuple.py's Model_RNTuple_Field.parent, and the file
    became unreadable).
    """
    path = os.path.join(tmp_path, "test.root")
    with uproot.recreate(path) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3, 4, 5], dtype=np.float32)}

    with uproot.update(path) as f:
        nt = f["mytuple"]
        nt.add_fields({"a": np.float32})
        assert [fr.field_name for fr in nt._cascading._existing_field_records] == [
            "x",
            "a",
        ]

        nt.add_fields({"b": np.float32})
        assert [fr.field_name for fr in nt._cascading._existing_field_records] == [
            "x",
            "a",
            "b",
        ]

        nt.add_fields({"c": np.float32})
        assert [fr.field_name for fr in nt._cascading._existing_field_records] == [
            "x",
            "a",
            "b",
            "c",
        ]

    with uproot.open(path) as f:
        nt = f["mytuple"]
        assert nt.keys() == ["x", "a", "b", "c"]
        arrays = nt.arrays()
        assert ak.all(arrays["x"] == np.array([1, 2, 3, 4, 5], dtype=np.float32))
        assert ak.all(arrays["a"] == np.zeros(5, dtype=np.float32))
        assert ak.all(arrays["b"] == np.zeros(5, dtype=np.float32))
        assert ak.all(arrays["c"] == np.zeros(5, dtype=np.float32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", path)
        assert reader.GetNEntries() == 5


def test_ntuple_extend_after_add_fields_column_order(tmp_path):
    """extend() after add_fields() must not scramble which data lands in which column.

    Regression test: extend() validates data.form == self._header._akform, but
    awkward Form equality for a RecordArray only checks the set of fields and
    their types, not their physical order. Every dict passed to extend() gets
    its keys alphabetically sorted by _regularize_input_type_to_awkward, but a
    header rebuilt by _load_existing_ntuple (which extend() triggers whenever
    a deferred column exists) reflects the true on-disk field order -- field
    insertion order, not alphabetical. For a schema like x (original), a, b
    (added together via add_fields), on-disk order is [x, a, b] but the data
    dict gets reordered to [a, b, x]. The form check passed (same fields, same
    types) while the column-buffer lookup below it is positional, keyed off
    self._header._column_keys built by walking the header's own field order.
    The result: 'x' silently received 'a's values, 'a' received 'b's, and 'b'
    received 'x's, from the very first post-add_fields extend() onward, with
    no exception raised. Confirmed independently against ROOT's own
    RNTupleReader, not just uproot's reader, since both were reading the same
    corrupted bytes.
    """
    path = os.path.join(tmp_path, "test.root")
    with uproot.recreate(path) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(path) as f:
        f["mytuple"].add_fields({"a": np.float32, "b": np.float32})

    with uproot.update(path) as f:
        f["mytuple"].extend(
            {
                "x": np.array([4.0, 5.0], dtype=np.float32),
                "a": np.array([40.0, 50.0], dtype=np.float32),
                "b": np.array([400.0, 500.0], dtype=np.float32),
            }
        )

    with uproot.open(path) as f:
        nt = f["mytuple"]
        assert ak.all(nt["x"].array() == np.array([1, 2, 3, 4, 5], dtype=np.float32))
        assert ak.all(nt["a"].array() == np.array([0, 0, 0, 40, 50], dtype=np.float32))
        assert ak.all(
            nt["b"].array() == np.array([0, 0, 0, 400, 500], dtype=np.float32)
        )

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", path)
        assert reader.GetNEntries() == 5
        entry = reader.CreateEntry()
        xs, as_, bs = [], [], []
        for i in range(reader.GetNEntries()):
            reader.LoadEntry(i, entry)
            xs.append(entry["x"])
            as_.append(entry["a"])
            bs.append(entry["b"])
        assert xs == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])
        assert as_ == pytest.approx([0.0, 0.0, 0.0, 40.0, 50.0])
        assert bs == pytest.approx([0.0, 0.0, 0.0, 400.0, 500.0])


def test_ntuple_extend_nested_field_order(tmp_path):
    """extend() must reorder nested record fields to match the header, not just top-level ones.

    Regression test: the fix for the top-level version of this bug
    (test_ntuple_extend_after_add_fields_column_order) only reordered
    top-level fields. A nested record whose fields are physically ordered
    differently than the header's form -- reachable directly via extend(),
    with no add_fields() involved -- still had its column buffers crossed,
    since awkward Form equality ignores field order at every level of
    nesting, not just the top. Manually confirmed independently against
    ROOT's own RNTupleReader (via RNTupleReader.GetView, which isn't used
    here because its cppyy binding has an unrelated interpreter-teardown
    crash in this environment).
    """
    path = os.path.join(tmp_path, "test.root")
    with uproot.recreate(path) as f:
        f["mytuple"] = ak.Array([{"p": {"a": 1.0, "b": 2.0}}])

    # nested record with fields in swapped physical order (b before a)
    inner = ak.zip({"b": np.array([4.0]), "a": np.array([3.0])})
    data = ak.zip({"p": inner}, depth_limit=1)
    assert data["p"].fields == ["b", "a"]

    with uproot.update(path) as f:
        f["mytuple"].extend(data)

    with uproot.open(path) as f:
        arrays = f["mytuple"].arrays()
        assert ak.all(arrays["p", "a"] == np.array([1.0, 3.0]))
        assert ak.all(arrays["p", "b"] == np.array([2.0, 4.0]))


def test_ntuple_accept_new_fields(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    # should raise without accept_new_fields
    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError, match="not in this RNTuple"):
            f["mytuple"].extend(
                {
                    "x": np.array([4, 5], dtype=np.float32),
                    "z": np.array([40, 50], dtype=np.int32),
                }
            )

    # with accept_new_fields=True - z backfilled with zeros, then user values
    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend(
            {
                "x": np.array([4, 5], dtype=np.float32),
                "z": np.array([40, 50], dtype=np.int32),
            },
            accept_new_fields=True,
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(nt["x"].array() == np.array([1, 2, 3, 4, 5], dtype=np.float32))
        assert ak.all(nt["z"].array() == np.array([0, 0, 0, 40, 50], dtype=np.int32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 5
        entry = reader.CreateEntry()
        vals = []
        for i in range(reader.GetNEntries()):
            reader.LoadEntry(i, entry)
            vals.append(entry["x"])
        assert vals == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_ntuple_add_subfield(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = ak.Array(
            [
                {"particle": {"pt": 1.0, "eta": 2.0}},
                {"particle": {"pt": 3.0, "eta": 4.0}},
            ]
        )

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"particle.phi": np.float32})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert "particle.phi" in nt.keys()
        assert ak.all(nt["particle.phi"].array() == np.zeros(2, dtype=np.float32))
        assert ak.all(nt["particle"].array().pt == np.array([1.0, 3.0]))
        assert ak.all(nt["particle"].array().phi == np.zeros(2, dtype=np.float32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 2


def test_ntuple_add_nested_subfield(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = ak.Array(
            [
                {"particle": {"track": {"pt": 1.0}}},
                {"particle": {"track": {"pt": 3.0}}},
            ]
        )

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"particle.track.phi": np.float32})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert "particle.track.phi" in nt.keys()
        assert ak.all(nt["particle.track.phi"].array() == np.zeros(2, dtype=np.float32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 2


def test_ntuple_add_subfield_nonexistent_parent(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError, match="not found"):
            f["mytuple"].add_fields({"nonexistent.phi": np.float32})


@pytest.mark.parametrize(
    "dotted_name",
    ["my_struct.new_field", "sub_struct.new_field", "sub_sub_struct.new_field"],
)
def test_ntuple_add_fields_encoding_error_checked_before_field_resolution(
    tmp_path, dotted_name
):
    """The column-encoding check must run before any per-field work, like extend().

    Regression test: add_fields() used to check self._column_encoding_error
    only inside the per-field loop, after already resolving the dotted-path
    parent (which can itself raise a ValueError, e.g. "has type ... and cannot
    be extended") and after appending the new field description to the
    shared, persistent footer.extension_field_record_frames list. That let a
    more specific but less fundamental error mask the real one, and left
    orphan field-description objects in memory on every failed call.

    Structs written by ROOT have C++ typenames and cannot have subfields added,
    but telling the user that is pointless when the whole file can't be extended
    regardless, so these names all raise the encoding error.
    """
    src = skhep_testdata.data_path("test_nested_structs_rntuple_v1-0-0-0.root")
    path = os.path.join(tmp_path, "test.root")
    shutil.copy(src, path)

    with uproot.update(path) as f:
        nt = f["ntuple"]
        before = len(nt._cascading._footer.extension_field_record_frames)
        with pytest.raises(ValueError, match="column encodings"):
            nt.add_fields({dotted_name: np.float32})
        after = len(nt._cascading._footer.extension_field_record_frames)
        assert after == before


def test_ntuple_add_subfield_to_collection(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"jets": ak.Array([[1.0, 2.0], [3.0]])}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError, match="'jets' has type .* cannot be extended"):
            f["mytuple"].add_fields({"jets.x": np.float32})


def test_ntuple_add_subfield_to_collection_of_records(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = ak.Array(
            {"jets": [[{"pt": 1.0, "eta": 0.0}], [{"pt": 2.0, "eta": 3.0}]]}
        )

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError, match="'jets' is not a record"):
            f["mytuple"].add_fields({"jets.phi": np.float32})


def test_ntuple_add_subfield_to_variant(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = ak.Array(
            {"variant": ak.Array([{"jet": {"pt": 1.0, "eta": 2.0}}, 2])}
        )

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(NotImplementedError, match="variant"):
            f["mytuple"].add_fields({"variant.jet.eta": np.float32})


def test_ntuple_access_variant_field_raises_clearly(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = ak.Array(
            {"variant": ak.Array([{"jet": {"pt": 1.0, "eta": 2.0}}, 2])}
        )

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(NotImplementedError, match="variant"):
            f["mytuple"]

    # reading it back with uproot.open() (rather than uproot.update()) works fine
    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert f["mytuple"].arrays().tolist() == [
            {"variant": {"jet": {"pt": 1.0, "eta": 2.0}}},
            {"variant": 2},
        ]


def test_ntuple_num_entries(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert nt.num_entries == 3


def test_ntuple_add_field_duplicate_after_extension(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"z": np.int32})

    # try to add z again - should fail even though it's an extension field
    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError, match="already exists"):
            f["mytuple"].add_fields({"z": np.float32})


def test_ntuple_add_fields_multi_cluster_groups(tmp_path):
    # multiple cluster groups (one cluster each) — should work
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["ntuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["ntuple"].extend({"x": np.array([4, 5, 6], dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["ntuple"].extend({"x": np.array([7, 8, 9], dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["ntuple"].add_fields({"y": np.int32})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert f["ntuple"].num_entries == 9
        assert np.all(f["ntuple"]["x"].array() == np.arange(1, 10, dtype=np.float32))
        assert np.all(f["ntuple"]["y"].array() == 0)


def test_ntuple_add_subfield_duplicate_name_raises(tmp_path):
    """add_fields() with a dotted path naming an already-existing sibling must raise, not corrupt.

    Regression test (reported after the initial round of fixes): the
    duplicate-field-name check only compared a new field's name against
    existing *top-level* fields, and a dotted path like "particle.pt" was
    never checked against "particle"'s existing children at all -- so
    add_fields({"particle.pt": ...}) when "particle.pt" already existed
    silently wrote a second field record with the same (parent_field_id,
    field_name). That's not just a harmless duplicate: it corrupts the
    RNTuple's field/ancestor bookkeeping badly enough that even a plain read
    of the file afterwards raised IndexError -- and since the footer had
    already reached disk by the time that surfaced, the file was left
    permanently unreadable (same failure class as the double-counting bug
    fixed in test_ntuple_multiple_add_fields_same_session).
    """
    path = os.path.join(tmp_path, "test.root")
    with uproot.recreate(path) as f:
        f["mytuple"] = ak.Array([{"particle": {"pt": 1.0}}] * 2)

    with uproot.update(path) as f:
        with pytest.raises(ValueError, match="already exists"):
            f["mytuple"].add_fields({"particle.pt": np.float32})

        # the file must still be usable after the rejected call
        f["mytuple"].add_fields({"particle.eta": np.float32})

    with uproot.open(path) as f:
        keys = f["mytuple"].keys()
        assert "particle.pt" in keys
        assert "particle.eta" in keys
        arrays = f["mytuple"].arrays()
        assert ak.all(arrays["particle", "pt"] == 1.0)
        assert ak.all(arrays["particle", "eta"] == 0.0)


def test_ntuple_add_subfield_duplicate_multilevel_name_raises(tmp_path):
    """The dotted-path duplicate check must also apply at 3+ levels of nesting."""
    path = os.path.join(tmp_path, "test.root")
    with uproot.recreate(path) as f:
        f["mytuple"] = ak.Array(
            [{"p1": {"track": {"pt": 1.0}}, "p2": {"track": {"pt": 2.0}}}] * 2
        )

    with uproot.update(path) as f:
        with pytest.raises(ValueError, match="already exists"):
            f["mytuple"].add_fields({"p2.track.pt": np.float32})

        # a legitimate new subfield under the same (correct) parent still works
        f["mytuple"].add_fields({"p2.track.phi": np.float32})

    with uproot.open(path) as f:
        keys = f["mytuple"].keys()
        assert "p2.track.phi" in keys
        assert "p1.track.phi" not in keys


def test_ntuple_add_subfield_ambiguous_name_raises(tmp_path):
    """A single-level dotted path ("parent.field") must not silently pick

    whichever same-named field happens to come first when the name exists at
    more than one nesting level.

    Regression test: the single-level branch matched a parent by bare name
    only, breaking on the first match in existing_field_records (a flat,
    depth-mixed, preorder-DFS list) -- with no check that the match was
    unambiguous. Given a top-level record "a" and an unrelated nested record
    "b.a", add_fields({"a.newsub": ...}) could silently attach "newsub" under
    "b.a" instead of top-level "a", with no exception.
    """
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = ak.Array([{"a": {"x": 1.0}, "b": {"a": {"y": 2.0}}}])

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError, match="ambiguous"):
            f["mytuple"].add_fields({"a.newsub": np.float32})

        # fully-qualified path still works to disambiguate
        f["mytuple"].add_fields({"b.a.newsub": np.float32})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        keys = f["mytuple"].keys()
        assert "b.a.newsub" in keys
        assert "a.newsub" not in keys


def test_ntuple_add_subfield_root_field_at_deeper_index(tmp_path):
    """Multi-level dotted-path resolution must not misidentify a non-root field as root.

    Regression test: is_root_field was computed as
    `fr.parent_field_id == 0 or fr.parent_field_id == i`. The `== 0` half is
    not a valid root signal on its own -- a field is top-level iff it's its
    own parent (self-referential), and any non-root field whose real parent
    happens to sit at index 0 (e.g. the second field of the very first
    top-level record) satisfies `parent_field_id == 0` too, without being
    root. Given top-level record "a" containing nested record "inner", and a
    separate, later, genuinely top-level record also named "inner" containing
    "child", resolving "inner.child.newsub" incorrectly matched the nested
    "inner" first and then failed to find "child" under it, raising a
    confusing "not found" error for a legitimate path.
    """
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = ak.Array(
            [{"a": {"inner": {"z": 1.0}}, "inner": {"child": {"y": 1.0}}}]
        )

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"inner.child.newsub": np.float32})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        keys = f["mytuple"].keys()
        assert "inner.child.newsub" in keys
        assert "a.inner.newsub" not in keys


def test_ntuple_add_field_duplicate_check_ignores_nested_names(tmp_path):
    """add_fields()'s duplicate-name check must only compare against top-level fields.

    Regression test: the check compared a new (always top-level) field name
    against every existing field's bare name, including nested ones. Adding a
    genuinely new top-level field "pt" was incorrectly rejected as "already
    exists" whenever any unrelated nested field also happened to be named
    "pt" (e.g. "particle.pt").
    """
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = ak.Array([{"particle": {"pt": 1.0}}])

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"pt": np.float32})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        keys = f["mytuple"].keys()
        assert "pt" in keys
        assert "particle.pt" in keys


def test_ntuple_update_root_written_file_opens(tmp_path):
    # ROOT-written files use UUID versions other than 1 — verify uproot.update can open them
    # This test uses a ROOT-written file and checks it can be accessed in update mode
    src = skhep_testdata.data_path("test_int_float_rntuple_v1-0-0-0.root")
    shutil.copy(src, os.path.join(tmp_path, "test.root"))

    # should not raise — previously failed with assert uuid_version == 1
    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        # accessing the ntuple should work (even though we can't extend ROOT-written files)
        with pytest.raises(ValueError, match="column encodings"):
            f["ntuple"].extend(
                {
                    "one_integers": np.array([1], dtype=np.int32),
                    "two_floats": np.array([1.0], dtype=np.float32),
                }
            )


def test_ntuple_hold_object_across_operations(tmp_path):
    # hold WritableNTuple object across add_fields and extend
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]  # hold the object
        nt.add_fields({"y": np.int32})
        nt.add_fields({"z": np.float64})
        nt.extend(
            {
                "x": np.array([4, 5], dtype=np.float32),
                "y": np.array([10, 20], dtype=np.int32),
                "z": np.array([1.1, 2.2], dtype=np.float64),
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert set(f["mytuple"].keys()) == {"x", "y", "z"}
        assert np.all(
            f["mytuple"]["x"].array() == np.array([1, 2, 3, 4, 5], dtype=np.float32)
        )
        assert np.all(
            f["mytuple"]["y"].array() == np.array([0, 0, 0, 10, 20], dtype=np.int32)
        )
        assert f["mytuple"]["z"].array().tolist() == [0, 0, 0, 1.1, 2.2]
