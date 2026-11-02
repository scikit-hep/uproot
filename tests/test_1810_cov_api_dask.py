# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot._dask

ak = pytest.importorskip("awkward")
dask = pytest.importorskip("dask")
pytest.importorskip("dask.array")
dask_awkward = pytest.importorskip("dask_awkward")


def _compute(lazy):
    return lazy.compute(scheduler="synchronous")


@pytest.fixture(scope="module")
def two_files(tmp_path_factory):
    """Two files with the same ``t``: ``x`` (flat) and ``j`` (jagged); ``a.root`` has 7 entries
    in two TBaskets and ``b.root`` has 5 in one, and only ``a.root`` has the extra ``only_a``.
    """
    tmp = tmp_path_factory.mktemp("dask")
    a, b = str(tmp / "a.root"), str(tmp / "b.root")
    jagged = ak.Array([[1.5], [], [2.5, 3.5], [4.5], [], [], [5.5, 6.5, 7.5]])
    with uproot.recreate(a) as file:
        tree = file.mktree("t", {"x": "int64", "j": "var * float64", "only_a": "int64"})
        for start, stop in ((0, 4), (4, 7)):
            tree.extend(
                {
                    "x": np.arange(start, stop),
                    "j": jagged[start:stop],
                    "only_a": np.arange(start, stop),
                }
            )
    with uproot.recreate(b) as file:
        tree = file.mktree("t", {"x": "int64", "j": "var * float64"})
        tree.extend({"x": np.arange(100, 105), "j": jagged[:5]})
    return a, b


EXPLICIT_STEPS = {"explicit": [[0, 3], [3, 7]]}


@pytest.mark.parametrize(
    ("open_files", "partitioning"),
    [
        pytest.param(True, {}, id="default"),
        pytest.param(False, {}, id="default delayed"),
        pytest.param(True, {"steps_per_file": 3}, id="steps_per_file"),
        pytest.param(False, {"steps_per_file": 3}, id="steps_per_file delayed"),
        pytest.param(True, {"step_size": 4}, id="step_size"),
        pytest.param(True, EXPLICIT_STEPS, id="explicit steps"),
        pytest.param(False, EXPLICIT_STEPS, id="explicit steps delayed"),
    ],
)
@pytest.mark.parametrize("library", ["np", "ak"])
def test_lazy_equals_eager(two_files, open_files, partitioning, library):
    a, b = two_files
    partitioning = dict(partitioning)
    explicit = partitioning.pop("explicit", None)
    if explicit is None:
        files = {a: "t", b: "t"}
    else:
        files = {
            a: {"object_path": "t", "steps": explicit},
            b: {"object_path": "t", "steps": [[0, 5]]},
        }

    expected = uproot.concatenate({a: "t", b: "t"}, ["x", "j"], library=library)
    lazy = uproot.dask(
        files,
        filter_name=["x", "j"],
        library=library,
        open_files=open_files,
        **partitioning,
    )

    if library == "np":
        assert set(lazy) == {"x", "j"}
        assert _compute(lazy["x"]).tolist() == expected["x"].tolist()
        jagged = _compute(lazy["j"])
        assert [list(row) for row in jagged] == [list(row) for row in expected["j"]]
    else:
        assert ak.to_list(_compute(lazy)) == ak.to_list(expected)


def test_slice_across_partitions_and_unknown_divisions(two_files):
    a, _ = two_files
    lazy = uproot.dask({a: "t"}, filter_name="x", steps_per_file=2, library="np")
    assert lazy["x"].chunks == ((4, 3),)
    assert _compute(lazy["x"][2:6]).tolist() == [2, 3, 4, 5]

    delayed = uproot.dask(
        {a: "t"}, filter_name="x", steps_per_file=2, library="ak", open_files=False
    )
    assert delayed.npartitions == 2
    assert not delayed.known_divisions
    assert _compute(delayed.x).tolist() == list(range(7))


def test_only_the_branches_common_to_all_files_are_kept(two_files):
    a, b = two_files
    lazy = uproot.dask({a: "t", b: "t"}, library="np")
    assert set(lazy) == {"x", "nj", "j"}
    assert set(uproot.dask({a: "t"}, library="np")) == {"x", "nj", "j", "only_a"}


def test_allow_missing_skips_the_file_without_the_tree(two_files):
    a, b = two_files
    files = {a: "t", b: "missing"}
    with pytest.raises(uproot.KeyInFileError, match="missing"):
        uproot.dask(files, library="np")
    lazy = uproot.dask(files, filter_name="x", library="np", allow_missing=True)
    assert _compute(lazy["x"]).tolist() == list(range(7))


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {"step_size": 2},
            TypeError,
            "partition sizes for 'files' is incompatible with 'step_size'",
        ),
        (
            {"steps_per_file": 2},
            TypeError,
            "partition sizes for 'files' is incompatible with 'steps_per_file'",
        ),
    ],
)
def test_partition_sizes_exclude_other_partitioning(two_files, kwargs, error, message):
    a, _ = two_files
    with pytest.raises(error, match=message):
        uproot.dask({a: {"object_path": "t", "steps": [[0, 7]]}}, **kwargs)


def test_partition_sizes_must_be_given_for_all_files(two_files):
    a, b = two_files
    with pytest.raises(
        TypeError, match="partition sizes for some but not all 'files' have been"
    ):
        uproot.dask({a: {"object_path": "t", "steps": [[0, 7]]}, b: "t"})


def test_known_base_form_needs_delayed_open(two_files):
    a, _ = two_files
    with uproot.open(a) as file:
        form = file["t"].arrays(["x"], entry_stop=1).layout.form
    with pytest.raises(
        TypeError, match="known_base_form must be None if open_files is True"
    ):
        uproot.dask({a: "t"}, known_base_form=form)


def test_pandas_is_not_implemented(two_files):
    a, _ = two_files
    with pytest.raises(NotImplementedError):
        uproot.dask({a: "t"}, library="pd")


def test_form_mapping_needs_awkward(two_files):
    a, _ = two_files
    with pytest.raises(NotImplementedError):
        uproot.dask(
            {a: "t"}, library="np", form_mapping=uproot._dask.TrivialFormMapping()
        )


@pytest.mark.parametrize("library", ["np", "ak"])
@pytest.mark.parametrize(
    ("options", "workers"),
    [
        ({}, (1, 1)),
        ({"num_workers": 3, "num_fallback_workers": 4}, (3, 4)),
    ],
)
def test_worker_options_reach_the_file(
    two_files, monkeypatch, library, options, workers
):
    a, _ = two_files
    regularize_object_path = uproot._util.regularize_object_path
    seen = []

    def spy(file_path, object_path, custom_classes, allow_missing, real_options):
        seen.append((real_options["num_workers"], real_options["num_fallback_workers"]))
        return regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, real_options
        )

    monkeypatch.setattr(uproot._util, "regularize_object_path", spy)
    lazy = uproot.dask(
        {a: "t"}, filter_name="x", library=library, open_files=False, **options
    )
    out = _compute(lazy["x"]) if library == "np" else _compute(lazy.x)
    assert out.tolist() == list(range(7))
    assert set(seen) == {workers}


@pytest.mark.parametrize("library", ["np", "ak"])
@pytest.mark.parametrize("steps", [[[0, 3], [3, 3], [3, 7]], [[0, 7], [7, 7]]])
def test_empty_explicit_steps_are_dropped(two_files, library, steps):
    a, _ = two_files
    lazy = uproot.dask(
        {a: {"object_path": "t", "steps": steps}}, filter_name="x", library=library
    )
    out = lazy["x"] if library == "np" else lazy
    if library == "np":
        assert sum(len(c) for c in out.chunks) == len(steps) - 1
        assert _compute(out).tolist() == list(range(7))
    else:
        assert out.npartitions == len(steps) - 1
        assert _compute(out.x).tolist() == list(range(7))


def test_explicit_steps_for_an_open_tree(two_files):
    a, _ = two_files
    with uproot.open(a) as file:
        # a TTree isn't hashable, but an iterable of them can be the key of "files"
        lazy = uproot.dask(
            {iter([file["t"]]): {"steps": [[0, 5], [5, 7]]}}, filter_name="x"
        )
        assert lazy.npartitions == 2
        assert _compute(lazy.x).tolist() == list(range(7))


OUT_OF_BOUNDS = r"explicit entry start \(0\) or stop \(100\) from uproot.dask 'files'"


@pytest.mark.parametrize("library", ["np", "ak"])
def test_explicit_steps_out_of_bounds_with_open_files(two_files, library):
    a, _ = two_files
    with pytest.raises(ValueError, match=OUT_OF_BOUNDS) as err:
        uproot.dask({a: {"object_path": "t", "steps": [[0, 100]]}}, library=library)
    assert "which has 7 entries" in str(err.value)


@pytest.mark.parametrize("library", ["np", "ak"])
def test_explicit_steps_out_of_bounds_without_open_files(two_files, library):
    a, _ = two_files
    lazy = uproot.dask(
        {a: {"object_path": "t", "steps": [[0, 100]]}},
        filter_name="x",
        library=library,
        open_files=False,
    )
    with pytest.raises(ValueError, match=OUT_OF_BOUNDS) as err:
        _compute(lazy["x"] if library == "np" else lazy.x)
    assert "which has 7 entries" in str(err.value)


def test_report_of_successful_reads(two_files):
    a, _ = two_files
    lazy, report = uproot.dask(
        {a: "t"}, filter_name="x", allow_read_errors_with_report=True
    )
    array, report = dask.compute(lazy, report, scheduler="synchronous")
    assert array.x.tolist() == list(range(7))
    assert len(report) == 1
    assert report.exception.tolist() == [None]
    assert report.message.tolist() == [None]
    assert report.args[0, 1:].tolist() == ["0", "7"]
    assert report.duration[0] >= 0
    # the tree's file was read
    assert report.performance_counters.num_requested_bytes[0] > 0


def test_trivial_form_mapping_is_the_default_read(two_files):
    a, b = two_files
    files = {a: "t", b: "t"}
    mapped = uproot.dask(
        files, filter_name=["x", "j"], form_mapping=uproot._dask.TrivialFormMapping()
    )
    plain = uproot.dask(files, filter_name=["x", "j"])
    assert ak.to_list(_compute(mapped)) == ak.to_list(_compute(plain))


def test_project_manually_reads_only_what_is_asked(two_files):
    a, _ = two_files
    lazy = uproot.dask({a: "t"}, filter_name=["x", "j"])
    layer = next(
        layer
        for layer in lazy.dask.layers.values()
        if hasattr(layer, "io_func") and hasattr(layer.io_func, "project_manually")
    )
    full = layer.io_func
    assert full.common_keys == frozenset({"x", "j"})

    projected = full.project_manually(frozenset({"x"}))
    assert projected.common_keys == frozenset({"x"})
    out = projected((0, 2, 5))
    assert out.layout.form.fields == ["x", "j"]
    assert out.x.tolist() == [2, 3, 4]
    with pytest.raises(Exception, match="excluded during a pre-run phase"):
        ak.to_list(out.j)


def test_form_mapping_base_class_has_no_projection():
    with pytest.raises(NotImplementedError):
        uproot._dask.UprootReadMixin().project_keys(frozenset())


def test_access_log_must_be_extendable(two_files):
    a, _ = two_files
    with uproot.open(a) as file:
        lazy = file["t"].arrays(filter_name="x", virtual=True, access_log=())
        with pytest.raises(ValueError, match=r"access_log=\(\) needs to implement"):
            ak.materialize(lazy.x)


class TestDaskArrayFromMap:
    @staticmethod
    def _add(x, y, offset=0, scale=1):
        return np.full(2, (x + y + offset) * scale)

    def test_several_iterables_are_paired(self):
        out = uproot._dask._dask_array_from_map(
            self._add, [1, 2], (x for x in [10, 20]), chunks=((2, 2),), dtype="i8"
        )
        assert _compute(out).tolist() == [11, 11, 22, 22]

    def test_args_and_kwargs_are_passed_through(self):
        out = uproot._dask._dask_array_from_map(
            self._add, [1, 2], args=(10,), chunks=((2, 2),), dtype="i8", offset=100
        )
        assert _compute(out).tolist() == [111, 111, 112, 112]

    def test_label_and_token_name_the_collection(self):
        out = uproot._dask._dask_array_from_map(
            self._add,
            [1],
            args=(10,),
            chunks=((2,),),
            dtype="i8",
            label="mine",
            token="abc",
        )
        assert out.name == "mine-abc"

    @pytest.mark.parametrize(
        ("func", "iterables", "kwargs", "message"),
        [
            (None, ([1],), {}, "`func` argument must be `callable`"),
            (abs, (1,), {}, "All elements of `iterables` must be Iterable, got "),
            (abs, (), {}, "`from_map` requires at least one Iterable input"),
            (abs, ([1], [1, 2]), {}, "All `iterables` must have the same length"),
            (abs, ([],), {}, "All `iterables` must have a non-zero length"),
            (
                abs,
                ([1], [2]),
                {"produces_tasks": True},
                "Multiple iterables not supported when produces_tasks=True",
            ),
        ],
    )
    def test_bad_inputs(self, func, iterables, kwargs, message):
        with pytest.raises(ValueError, match=message):
            uproot._dask._dask_array_from_map(
                func, *iterables, chunks=((1,),), dtype="i8", **kwargs
            )


def test_lazy_inputs_dict():
    lazy = uproot._dask._LazyInputsDict(["a", "b"])
    assert len(lazy) == 2
    assert list(lazy.keys()) == [(0,), (1,)]
    assert list(lazy) == ["a", "b"]  # as dask's iteration over values expects
    assert lazy[(1,)] == "b"
    assert (1,) in lazy
    assert (2,) not in lazy
    assert (-1,) not in lazy
    assert "a" not in lazy
    assert ("a",) not in lazy


def test_packed_arg_callable():
    call = uproot._dask._PackedArgCallable(lambda *args: args, args=[3], packed=True)
    assert call((1, 2)) == (1, 2, 3)
    unpacked = uproot._dask._PackedArgCallable(
        lambda x, **kwargs: (x, kwargs), kwargs={"y": 2}
    )
    assert unpacked((1, 2)) == ((1, 2), {"y": 2})
    assert uproot._dask._PackedArgCallable(lambda x: x)(5) == 5


class TestCoerce:
    def test_no_expected_dtype_passes_through(self):
        array = np.arange(3)
        assert uproot._dask._coerce_numpy_chunk(array, None) is array

    def test_same_structured_dtype_passes_through(self):
        array = np.zeros(2, dtype=[("a", "i4")])
        assert uproot._dask._coerce_numpy_chunk(array, array.dtype) is array

    def test_structured_fields_are_coerced_one_by_one(self):
        array = np.array([(1, 2.5), (3, 4.5)], dtype=[("a", "i4"), ("b", "f8")])
        expected = np.dtype([("a", "i8"), ("b", "O")])
        out = uproot._dask._coerce_numpy_chunk(array, expected)
        assert out.dtype == expected
        assert out.tolist() == [(1, 2.5), (3, 4.5)]

    def test_plain_dtype(self):
        out = uproot._dask._coerce_numpy_chunk(np.arange(2, dtype="i4"), np.dtype("f8"))
        assert out.dtype == np.dtype("f8")
        assert out.tolist() == [0.0, 1.0]

    def test_same_dtype_passes_through(self):
        array = np.arange(2, dtype="i4")
        assert uproot._dask._coerce_field(array, np.dtype("i4")) is array

    def test_subarray_of_the_right_shape(self):
        array = np.arange(4, dtype="i4").reshape(2, 2)
        same = uproot._dask._coerce_field(array, np.dtype(("i4", (2,))))
        assert same is array
        wider = uproot._dask._coerce_field(array, np.dtype(("i8", (2,))))
        assert wider.dtype == np.dtype("i8")
        assert wider.tolist() == array.tolist()

    def test_to_object_rows(self):
        array = np.arange(4).reshape(2, 2)
        out = uproot._dask._coerce_field(array, np.dtype("O"))
        assert out.shape == (2,)
        assert [row.tolist() for row in out] == [[0, 1], [2, 3]]

    def test_masked_to_object_rows(self):
        array = np.ma.MaskedArray(
            np.arange(6.0).reshape(3, 2),
            mask=[[False, False], [True, False], [True, True]],
        )
        out = uproot._dask._coerce_field(array, np.dtype("O"))
        assert out[0].tolist() == [0.0, 1.0]
        assert out[1].dtype == np.dtype("O")
        assert out[1].tolist() == [None, 3.0]
        assert out[2] is None

    def test_masked_to_object_scalars(self):
        array = np.ma.MaskedArray([1, 2, 3], mask=[False, True, False])
        out = uproot._dask._coerce_field(array, np.dtype("O"))
        assert out.tolist() == [1, None, 3]

    def test_to_object_scalars(self):
        out = uproot._dask._coerce_field(np.arange(2), np.dtype("O"))
        assert out.dtype == np.dtype("O")
        assert out.tolist() == [0, 1]
        assert uproot._dask._coerce_field(out, np.dtype("O")) is out


def test_fixed_size_arrays_of_strided_objects():
    """``TLorentzVector[][5]``: a list per entry of arrays of 5 objects."""
    path = skhep_testdata.data_path("uproot-issue513.root")
    name = "Jet.TrimmedP4[5]"
    lazy = uproot.dask(
        f"{path}:Delphes", filter_name=r"/^Jet\.TrimmedP4\[5\]$/", open_files=False
    )
    out = _compute(lazy)[name]
    assert str(out.type) == (
        "10 * var * 5 * TLorentzVector[fP: TVector3[fX: float64, fY: float64, "
        "fZ: float64], fE: float64]"
    )
    assert ak.num(out, axis=1).tolist() == [8, 6, 5, 4, 5, 5, 6, 6, 6, 9]


def test_rntuple_as_numpy_dtype():
    path = skhep_testdata.data_path("test_int_float_rntuple_v1-0-0-0.root")
    with uproot.open(path) as file:
        dtype = uproot._dask._branch_or_field_numpy_dtype(file["ntuple"])
    assert dtype == np.dtype([("one_integers", "<i4"), ("two_floats", "<f4")])


def test_field_missing_from_its_form():
    class Field(uproot._dask.HasFields):
        def to_akform(self):
            form = ak.forms.RecordForm([ak.forms.NumpyForm("int64")], ["a"])
            return form, ["b"]

    with pytest.raises(KeyError, match="field 'b' not found in form"):
        uproot._dask._branch_or_field_numpy_dtype(Field())


def test_field_inside_an_option():
    """As ROOT writes ``std::optional<std::pair<int, float>>``: an option has no size to add."""

    class Field(uproot._dask.HasFields):
        def to_akform(self):
            pair = ak.forms.RecordForm([ak.forms.NumpyForm("int32")], ["0"])
            form = ak.forms.RecordForm([ak.forms.IndexedOptionForm("i64", pair)], ["o"])
            return form, ["o", "0"]

    assert uproot._dask._branch_or_field_numpy_dtype(Field()) == np.dtype("int32")


def test_rntuple_fields_of_regular_records_as_numpy(tmp_path):
    path = str(tmp_path / "regular.root")
    records = ak.Array(
        [
            [{"x": 1, "y": 2.0}, {"x": 3, "y": 4.0}],
            [{"x": 5, "y": 6.0}, {"x": 7, "y": 8.0}],
        ]
    )
    with uproot.recreate(path) as file:
        file.mkrntuple("n", {"regular": ak.to_regular(records)})

    lazy = uproot.dask({path: "n"}, library="np")
    assert set(lazy) == {"regular", "regular.x", "regular.y"}
    assert lazy["regular.x"].shape == (2, 2)
    assert _compute(lazy["regular.x"]).tolist() == [[1, 3], [5, 7]]
    assert _compute(lazy["regular.y"]).tolist() == [[2.0, 4.0], [6.0, 8.0]]
    assert _compute(lazy["regular"]).tolist() == [
        [(1, 2.0), (3, 4.0)],
        [(5, 6.0), (7, 8.0)],
    ]


def test_rntuple_union_field_lazy_equals_eager():
    path = skhep_testdata.data_path("test_stl_containers_rntuple_v1-0-0-0.root")
    name = "variant_int32_string"
    with uproot.open(path) as file:
        expected = file["ntuple"].arrays(filter_name=name)
    lazy = uproot.dask({path: "ntuple"}, filter_name=name)
    assert ak.to_list(_compute(lazy)) == ak.to_list(expected)


def test_dask_write_without_a_prefix(two_files, tmp_path):
    a, _ = two_files
    lazy = uproot.dask(
        {a: "t"}, filter_name=["x", "only_a"], steps_per_file=2, open_files=False
    )
    with dask.config.set(scheduler="synchronous"):
        uproot.dask_write(lazy, str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["part0.root", "part1.root"]
    out = uproot.concatenate(
        [str(tmp_path / "part0.root"), str(tmp_path / "part1.root")],
        library="np",
    )
    assert out["x"].tolist() == list(range(7))
    assert out["only_a"].tolist() == list(range(7))
