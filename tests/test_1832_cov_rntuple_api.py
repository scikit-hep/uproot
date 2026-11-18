# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import io
import re

import numpy as np
import pytest

import uproot
from uproot.behaviors.RNTuple import concatenate, iterate

ak = pytest.importorskip("awkward")


@pytest.fixture
def ntuple_path(tmp_path):
    """A small RNTuple with nested records, lists, and documented fields."""
    path = tmp_path / "nested.root"
    rec = ak.Array(
        [
            {"a": 1, "inner": {"deep": 1.5}},
            {"a": 2, "inner": {"deep": 2.5}},
            {"a": 3, "inner": {"deep": 3.5}},
        ]
    )
    vec = ak.Array([[{"p": 1}], [], [{"p": 2}, {"p": 3}]])
    data = ak.with_field(ak.Array({"x": [10, 20, 30]}), rec, "rec")
    data = ak.with_field(data, vec, "vec")
    data = ak.with_parameter(data, "__doc__", "Some entries")
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", data, description="The description")
    return path


@pytest.fixture
def ntuple(ntuple_path):
    with uproot.open(ntuple_path) as file:
        yield file["ntuple"]


ALL_KEYS = ["x", "rec", "rec.a", "rec.inner", "rec.inner.deep", "vec", "vec.p"]


def _write_x(path, values):
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", {"x": np.asarray(values, dtype=np.int64)})
    return str(path)


@pytest.fixture
def three_files(tmp_path):
    return [
        _write_x(tmp_path / "first.root", [0, 1, 2]),
        _write_x(tmp_path / "second.root", [3, 4, 5, 6]),
        _write_x(tmp_path / "third.root", [7, 8]),
    ]


@pytest.mark.parametrize(
    ("entry_start", "entry_stop"),
    [
        (None, None),
        # spans a whole file in the middle, which is read completely
        (1, 8),
        # leaves the last files out
        (0, 2),
        # leaves the first files out
        (7, None),
        # only in the middle file
        (4, 6),
        # negative positions count from the end
        (-2, None),
    ],
)
def test_concatenate_entry_range(three_files, entry_start, entry_stop):
    out = concatenate(
        {path: "ntuple" for path in three_files},
        entry_start=entry_start,
        entry_stop=entry_stop,
    )
    assert out.x.to_list() == list(range(9))[entry_start:entry_stop]


@pytest.fixture
def one_file_without_ntuple(tmp_path):
    with_ntuple = _write_x(tmp_path / "with.root", [1, 2, 3])
    without_ntuple = tmp_path / "without.root"
    with uproot.recreate(without_ntuple) as file:
        file["other"] = "a TObjString"
    return {with_ntuple: "ntuple", str(without_ntuple): "ntuple"}


def test_iterate_allow_missing_object(one_file_without_ntuple):
    out = list(iterate(one_file_without_ntuple, step_size=2, allow_missing=True))
    assert [chunk.x.to_list() for chunk in out] == [[1, 2], [3]]


def test_concatenate_allow_missing_object(one_file_without_ntuple):
    out = concatenate(one_file_without_ntuple, allow_missing=True)
    assert out.x.to_list() == [1, 2, 3]


@pytest.mark.parametrize(
    "read",
    [
        lambda files: list(iterate(files, step_size=2)),
        lambda files: concatenate(files),
    ],
)
def test_missing_object_is_an_error_by_default(one_file_without_ntuple, read):
    with pytest.raises(uproot.KeyInFileError, match="ntuple"):
        read(one_file_without_ntuple)


@pytest.fixture
def one_file_without_subfield(tmp_path):
    """Both files have a record, but only the first has the subfield ``y``."""
    paths = {}
    for name, fields in (("with", ("x", "y")), ("without", ("x",))):
        path = str(tmp_path / f"{name}.root")
        with uproot.recreate(path) as file:
            file.mkrntuple(
                "ntuple", {"rec": ak.Array([{field: 1 for field in fields}] * 2)}
            )
        paths[path] = "ntuple"
    return paths


def _select_records_with_y(field):
    # a filter on an RField can look at its subfields, which raises if they are missing
    return field.name != "rec" or field["y"].typename == "std::int64_t"


def test_iterate_allow_missing_field(one_file_without_subfield):
    out = list(
        iterate(
            one_file_without_subfield,
            filter_field=_select_records_with_y,
            allow_missing=True,
        )
    )
    assert [chunk.to_list() for chunk in out] == [[{"rec": {"x": 1, "y": 1}}] * 2]


def test_concatenate_allow_missing_field(one_file_without_subfield):
    out = concatenate(
        one_file_without_subfield,
        filter_field=_select_records_with_y,
        allow_missing=True,
    )
    assert out.to_list() == [{"rec": {"x": 1, "y": 1}}] * 2


@pytest.mark.parametrize(
    "read",
    [
        lambda files: list(iterate(files, filter_field=_select_records_with_y)),
        lambda files: concatenate(files, filter_field=_select_records_with_y),
    ],
)
def test_missing_field_is_an_error_by_default(one_file_without_subfield, read):
    with pytest.raises(uproot.KeyInFileError, match="not found"):
        read(one_file_without_subfield)


def test_path_of_an_rntuple_is_a_dot(ntuple):
    assert ntuple.path == "."
    assert ntuple["rec"]["inner"]["deep"].path == "rec.inner.deep"


@pytest.mark.parametrize(
    "method",
    [
        lambda ntuple, **kwargs: ntuple.arrays("x", **kwargs),
        lambda ntuple, **kwargs: list(ntuple.iterate("x", **kwargs)),
        lambda ntuple, **kwargs: ntuple.num_entries_for("1 MB", "x", **kwargs),
    ],
)
def test_expressions_and_filter_name_are_exclusive(ntuple, method):
    with pytest.raises(ValueError, match="Expressions are not supported yet"):
        method(ntuple, filter_name="rec")


def test_expressions_act_as_filter_name(ntuple):
    assert ntuple.arrays("x").fields == ["x"]
    assert [chunk.fields for chunk in ntuple.iterate("x")] == [["x"]]
    assert ntuple.num_entries_for("1 kB", "x") == ntuple.num_entries_for(
        "1 kB", filter_name="x"
    )
    # 8 bytes per entry, a 3 entry cluster
    assert ntuple.num_entries_for(80, "x") == 10


def test_num_entries_for_nothing_selected(ntuple):
    assert ntuple.num_entries_for("1 MB", filter_name="does_not_exist") is None


@pytest.mark.parametrize("filter_name", ["x", "vec"])
def test_num_entries_for_an_empty_entry_range(ntuple, filter_name):
    # no clusters are in the range, so there is no size to go by
    assert (
        ntuple.num_entries_for(
            "1 MB", filter_name=filter_name, entry_start=2, entry_stop=2
        )
        == 1
    )


@pytest.mark.parametrize(
    ("interpreter", "message"),
    [
        ("gpu", "Backend cpu GDS support not implemented."),
        ("unknown", "Backend cpu not implemented."),
    ],
)
def test_unsupported_interpreter(ntuple, interpreter, message):
    with pytest.raises(NotImplementedError, match=message):
        ntuple.arrays(interpreter=interpreter, backend="cpu")


def test_array_protocol_of_a_field(ntuple):
    out = np.asarray(ntuple["x"])
    assert out.dtype == np.dtype("int64")
    assert out.tolist() == [10, 20, 30]

    converted = np.asarray(ntuple["x"], dtype=np.float32)
    assert converted.dtype == np.dtype("float32")
    assert converted.tolist() == [10.0, 20.0, 30.0]


def test_array_protocol_of_an_rntuple(ntuple):
    # An RNTuple is a table of fields, not one array
    assert ntuple.__array__()["x"].tolist() == [10, 20, 30]
    with pytest.raises(ValueError, match="__array__"):
        np.asarray(ntuple)
    wrapped = ntuple.__array__(dtype=object)
    assert wrapped.shape == ()
    assert wrapped.item()["x"].tolist() == [10, 20, 30]


def test_iterate_report(three_files):
    out = list(
        iterate(
            {path: "ntuple" for path in three_files[:2]},
            step_size=2,
            report=True,
        )
    )
    assert [chunk.x.to_list() for chunk, _ in out] == [[0, 1], [2], [3, 4], [5, 6]]
    assert [
        (report.global_entry_start, report.global_entry_stop) for _, report in out
    ] == [(0, 2), (2, 3), (3, 5), (5, 7)]
    assert [(report.tree_entry_start, report.tree_entry_stop) for _, report in out] == [
        (0, 2),
        (2, 3),
        (0, 2),
        (2, 4),
    ]


def test_items(ntuple):
    assert [
        (name, field.typename) for name, field in ntuple.items(recursive=False)
    ] == [
        ("x", "std::int64_t"),
        ("rec", ""),
        ("vec", ""),
    ]
    assert ntuple.items(filter_name="deep")[0][1] is ntuple["rec.inner.deep"]


def test_values(ntuple):
    assert [field.name for field in ntuple.values(recursive=False)] == [
        "x",
        "rec",
        "vec",
    ]
    assert [field.name for field in ntuple["rec"].values()] == ["a", "inner", "deep"]


def test_filter_branch_is_deprecated(ntuple):
    with pytest.warns(DeprecationWarning, match="filter_branch kwarg should not"):
        assert ntuple.keys(filter_branch=lambda field: field.name == "x") == ["x"]


def test_filter_field_none_selects_everything(ntuple):
    assert ntuple.keys(filter_field=None) == ALL_KEYS


def test_filter_field_must_be_callable(ntuple):
    with pytest.raises(TypeError, match="filter_field must be None or a function"):
        ntuple.keys(filter_field="x")


def test_typenames(ntuple):
    assert ntuple.typenames(filter_name="x") == {"x": "std::int64_t"}
    assert ntuple.typenames(filter_typename="double") == {"rec.inner.deep": "double"}
    assert dict(ntuple["rec"].itertypenames(recursive=False)) == {
        "a": "std::int64_t",
        "inner": "",
    }


def test_ipython_key_completions(ntuple):
    assert list(ntuple._ipython_key_completions_()) == ALL_KEYS


def test_fields_by_position(ntuple):
    assert [field.name for field in ntuple] == ["x", "rec", "vec"]
    assert ntuple[1] is ntuple["rec"]
    assert ntuple["rec"][0] is ntuple["rec"]["a"]
    assert ntuple[-1].name == "vec"


@pytest.mark.parametrize(
    "where",
    ["rec.inner.deep", "rec/inner/deep", "rec\\inner\\deep", "rec..inner.deep", "deep"],
)
def test_fields_by_name(ntuple, where):
    assert ntuple[where] is ntuple["rec"]["inner"]["deep"]


def test_recursive_lookup_of_a_later_field(ntuple):
    # the first fields are searched without success
    assert ntuple["p"].path == "vec.p"
    # and the result of a search is not confused with a parent's fields
    assert ntuple["a"] is ntuple["rec"]["a"]
    assert ntuple["rec"]["deep"] is ntuple["rec"]["inner"]["deep"]


@pytest.mark.parametrize(
    "where",
    ["does_not_exist", "rec.does_not_exist", ".deep"],
)
def test_missing_field_lookup(ntuple, ntuple_path, where):
    with pytest.raises(uproot.KeyInFileError, match="not found") as excinfo:
        ntuple[where]
    assert excinfo.value.key == where
    assert excinfo.value.file_path.endswith(ntuple_path.name)


def test_field_lookup_with_a_bad_type(ntuple):
    with pytest.raises(TypeError, match="where must be an integer or a string"):
        ntuple[1.5]


def test_show(ntuple):
    stream = io.StringIO()
    ntuple.show(stream=stream)
    assert stream.getvalue() == """ntuple (ROOT::RNTuple)
Description: The description
├─ x (std::int64_t)
├─ rec ()
│  ├─ a (std::int64_t)
│  └─ inner ()
│     └─ deep (double)
└─ vec ()
   └─ p (std::int64_t)
"""


def test_show_not_recursive_and_narrow(ntuple):
    stream = io.StringIO()
    ntuple.show(stream=stream, recursive=False, max_width=12)
    assert stream.getvalue().splitlines() == [
        "ntuple (ROOT",
        "Description:",
        "├─ x (std::i",
        "├─ rec ()",
        "└─ vec ()",
    ]


def test_show_a_field_with_nothing_selected(ntuple):
    stream = io.StringIO()
    ntuple["rec"].show(stream=stream, filter_name="does_not_exist")
    assert stream.getvalue() == "rec ()\n"


def test_show_field_descriptions(tmp_path):
    form = ak.forms.RecordForm(
        [
            ak.forms.NumpyForm("int32", parameters={"__doc__": "the first"}),
            ak.forms.RecordForm(
                [ak.forms.NumpyForm("int32", parameters={"__doc__": "the inner"})],
                ["c"],
                parameters={"__doc__": "the second"},
            ),
            ak.forms.NumpyForm("int32"),
        ],
        ["a", "b", "d"],
    )
    path = tmp_path / "documented.root"
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", form)

    stream = io.StringIO()
    with uproot.open(path) as file:
        file["ntuple"].show(stream=stream)
    assert stream.getvalue() == """ntuple (ROOT::RNTuple)
├─ a (std::int32_t)
│  Description: the first
├─ b ()
│  Description: the second
│  └─ c (std::int32_t)
│     Description: the inner
└─ d (std::int32_t)
"""


def test_source(ntuple, ntuple_path):
    assert ntuple.source.file_path.endswith(ntuple_path.name)


def test_source_of_a_detached_rntuple(ntuple_path):
    with uproot.open(ntuple_path) as file:
        # an RNTuple read as part of another object is on a detached file
        key = file.key("ntuple")
        chunk, cursor = key.get_uncompressed_chunk_cursor()
        detached = file.file.class_named("ROOT::RNTuple").read(
            chunk,
            cursor,
            {"breadcrumbs": (), "TKey": key},
            file.file,
            file.file.detached,
            None,
        )
        assert detached.source is None


def test_reprs(ntuple):
    assert re.fullmatch(
        r"<ROOT::RNTuple 'ntuple' \(3 fields\) at 0x[0-9a-f]{12,}>", repr(ntuple)
    )
    assert re.fullmatch(
        r"<RField 'rec' \(2 subfields\) in RNTuple 'ntuple' at 0x[0-9a-f]{12,}>",
        repr(ntuple["rec"]),
    )
    assert re.fullmatch(
        r"<RField 'x' in RNTuple 'ntuple' at 0x[0-9a-f]{12,}>", repr(ntuple["x"])
    )


def test_repr_without_fields(tmp_path):
    path = tmp_path / "empty.root"
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", ak.forms.RecordForm([], []))
    with uproot.open(path) as file:
        assert re.fullmatch(
            r"<ROOT::RNTuple 'ntuple' at 0x[0-9a-f]{12,}>", repr(file["ntuple"])
        )


def test_field_index(ntuple):
    assert [field.index for field in ntuple.fields] == [0, 1, 2]
    assert ntuple["rec"]["inner"].index == 1


def test_member_that_shadows_an_inherited_member(tmp_path):
    # ":_N" is how a base class is stored
    path = tmp_path / "file.root"
    data = ak.Array([{"member": 1, ":_0": {"member": 2}}] * 2)
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", {"c": data})
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        assert ntuple.keys() == ["c", "c.member", "c.::member"]
        assert ntuple.arrays().c.to_list() == [{"member": 1, "::member": 2}] * 2


def test_field_names(ntuple):
    assert ntuple.field_names is ntuple.field_names
    assert ntuple.field_names[:2] == ["x", "rec"]
