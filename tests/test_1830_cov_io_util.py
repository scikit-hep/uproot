# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os
import pathlib
import re
import sys
import types
from urllib.request import url2pathname

import awkward
import numpy
import pytest
import skhep_testdata

import uproot
import uproot._util


def test_ensure_str():
    assert uproot._util.ensure_str("x") == "x"
    assert uproot._util.ensure_str(b"x\xff") == "x\udcff"
    with pytest.raises(TypeError, match="expected a string, not <class 'int'>"):
        uproot._util.ensure_str(1)


def test_ensure_numpy_of_awkward_content():
    content = awkward.contents.NumpyArray(numpy.array([1, 2, 3]))
    out = uproot._util.ensure_numpy(content)
    assert isinstance(out, numpy.ndarray)
    assert out.tolist() == [1, 2, 3]


def test_regex_flags():
    assert uproot._util._regularize_filter_regex_flags("") == 0
    assert uproot._util._regularize_filter_regex_flags("iLmsux") == (
        re.IGNORECASE | re.LOCALE | re.MULTILINE | re.DOTALL | re.UNICODE | re.VERBOSE
    )
    assert uproot._util._regularize_filter_regex_flags("z") == 0  # not a flag


def test_filter_flags_are_applied():
    assert uproot._util.regularize_filter("/AB/i")("abc") is True
    assert uproot._util.regularize_filter("/a.b/s")("a\nb") is True
    assert uproot._util.regularize_filter("/a.b/")("a\nb") is False
    assert uproot._util.regularize_filter("/a b/x")("ab") is True


def test_regularize_filter_none_and_error():
    assert uproot._util.regularize_filter(None) is uproot._util.no_filter
    assert uproot._util.no_filter("anything") is True
    with pytest.raises(TypeError, match="filter must be None, callable.*not 3"):
        uproot._util.regularize_filter(3)
    with pytest.raises(TypeError, match="filter must be None, callable.*not b'x'"):
        uproot._util.regularize_filter(b"x")


def test_regularize_rename_none_and_callable():
    assert uproot._util.regularize_rename(None) is uproot._util.no_rename
    assert uproot._util.no_rename("x") == "x"
    assert uproot._util.regularize_rename(str.upper) is str.upper


@pytest.mark.parametrize("rename", ["/(.*)_pt/pt_\\1/", "s/(.*)_pt/pt_\\1/"])
def test_regularize_rename_regex(rename):
    function = uproot._util.regularize_rename(rename)
    assert function("muon_pt") == "pt_muon"
    assert function("muon_eta") == "muon_eta"


def test_regularize_rename_regex_flags():
    assert uproot._util.regularize_rename("/PT/x/i")("muon_pt") == "muon_x"


def test_regularize_rename_dict():
    function = uproot._util.regularize_rename({"a": "b"})
    assert function("a") == "b"
    assert function("c") == "c"


def test_regularize_rename_rules_stop_at_the_first_that_matches():
    function = uproot._util.regularize_rename(["/a/b/", "/b/c/", "/D/e/i"])
    assert function("a") == "b"
    assert function("b") == "c"
    assert function("d") == "e"
    assert function("z") == "z"


@pytest.mark.parametrize("rename", ["no-slashes", ["/a/b/", "no-slashes"]])
def test_regularize_rename_bad_regex(rename):
    with pytest.raises(TypeError, match="must be in '/from/to/' form"):
        uproot._util.regularize_rename(rename)


@pytest.mark.parametrize("rename", [3, b"/a/b/", ["/a/b/", 3]])
def test_regularize_rename_bad_type(rename):
    with pytest.raises(TypeError, match="rename must be None, callable"):
        uproot._util.regularize_rename(rename)


def test_canonical_path_unknown_protocol_is_kept():
    url = "not-a-known-protocol://host/file.root"
    assert uproot._util.canonical_path(url) == url


def test_canonical_path_windows_drive_in_file_url(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    expected = os.path.realpath(url2pathname("C:/dir/file.root"))
    assert uproot._util.canonical_path("file://C:/dir/file.root") == expected


def test_canonical_path_other_host_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    url = "file://server/dir/file.root"
    assert uproot._util.canonical_path(url) == url


def test_file_path_to_source_class_bad_type():
    with pytest.raises(
        TypeError, match="file_path is not a string or file-like object: 3"
    ):
        uproot._util.file_path_to_source_class(3, {"handler": None})


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1 kB", 10**3),
        ("1 MB", 10**6),
        ("1 GB", 10**9),
        ("1 TB", 10**12),
        ("1 PB", 10**15),
        ("1 EB", 10**18),
        ("1 ZB", 1000.0**7),
        ("1 YB", 1000.0**8),
        ("1 kiB", 2**10),
        ("1 MiB", 2**20),
        ("1 GiB", 2**30),
        ("1 TiB", 2**40),
        ("1 PiB", 2**50),
        ("1 EiB", 2**60),
        ("1 ZiB", 2**70),
        ("1 YiB", 2**80),
        ("1.5 b", 1),
        ("2e3 B", 2000),
    ],
)
def test_memory_size_units(text, expected):
    assert uproot._util.memory_size(text) == int(expected)


def test_memory_size_of_a_number():
    assert uproot._util.memory_size(123) == 123
    assert uproot._util.memory_size(numpy.int64(123)) == 123


@pytest.mark.parametrize("data", ["100 parsecs", 1.5, None])
def test_memory_size_error_message(data):
    with pytest.raises(TypeError, match="memory size string with units"):
        uproot._util.memory_size(data)
    with pytest.raises(TypeError, match="^my message$"):
        uproot._util.memory_size(data, "my message")


@pytest.mark.parametrize(
    "entry_start, entry_stop, expected",
    [
        (12, 18, [list(range(12, 18))]),  # inside one basket
        (0, 10, [list(range(10))]),  # exactly one basket
        (5, 25, [list(range(5, 10)), list(range(10, 20)), list(range(20, 25))]),
        (10, 30, [[], list(range(10, 20)), list(range(20, 30))]),  # two whole baskets
    ],
)
def test_trim_final(entry_start, entry_stop, expected):
    baskets = {
        0: numpy.arange(0, 10),
        1: numpy.arange(10, 20),
        2: numpy.arange(20, 30),
    }
    trimmed = uproot._util.trim_final(
        baskets, entry_start, entry_stop, [0, 10, 20, 30], "np", None
    )
    assert [x.tolist() for x in trimmed if len(x) > 0] == [
        x for x in expected if len(x) > 0
    ]
    assert numpy.concatenate(trimmed).tolist() == list(range(entry_start, entry_stop))


def test_trim_final_skips_baskets_outside_the_range():
    baskets = {
        0: numpy.arange(0, 10),
        1: numpy.arange(10, 20),
        2: numpy.arange(20, 30),
        3: numpy.arange(30, 40),
    }
    trimmed = uproot._util.trim_final(baskets, 12, 15, [0, 10, 20, 30, 40], "np", None)
    assert [x.tolist() for x in trimmed] == [[12, 13, 14]]


def test_awkward_form_of_structured_dtype():
    dtype = numpy.dtype([("x", ">i4"), ("y", ">f8")])
    form = uproot._util.awkward_form(dtype, None, {})
    assert form == awkward.forms.RecordForm(
        [awkward.forms.NumpyForm("int32"), awkward.forms.NumpyForm("float64")],
        ["x", "y"],
    )


def test_awkward_form_of_iter_unrecognized():
    with pytest.raises(RuntimeError, match="unrecognized form: <class 'str'>"):
        uproot._util.awkward_form_of_iter(awkward, "not a form")


def test_ak_add_doc_parameters_of_something_else():
    assert uproot._util.ak_add_doc_parameters(object(), None) == {}


@pytest.mark.parametrize(
    "a, b, distance",
    [
        ("ab", "ba", 1),  # transposition
        ("abcd", "acbd", 1),
        ("kitten", "sitting", 3),
        ("ab", "", 2),
    ],
)
def test_edit_distance(a, b, distance):
    assert uproot._util._edit_distance(a, b) == distance
    assert uproot._util._edit_distance(b, a) == distance


@pytest.mark.parametrize(
    "a, b, distance",
    [
        ("abc", "abc", 0),
        ("abc", "abC", 0.5),  # capitalization is a middling penalty
        ("abcd", "abdc", 1),  # transposition
        ("ab", "BA", 1.5),  # transposition and capitalization
        ("abc", "xyz", 6),
    ],
)
def test_damerau_levenshtein(a, b, distance):
    assert uproot._util.damerau_levenshtein(a, b) == distance


def test_regularize_steps():
    assert uproot._util.regularize_steps([0, 10, 20]) == [[0, 10], [10, 20]]
    assert uproot._util.regularize_steps([[0, 5], [8, 8]]) == [[0, 5], [8, 8]]


@pytest.mark.parametrize(
    "steps, error, message",
    [
        ({"a": 1}, TypeError, "must be an iterable of integer offsets or start-stop"),
        ([0.5, 1.5], TypeError, "must be an iterable of integer offsets or start-stop"),
        (numpy.array([], dtype=int), ValueError, "must be non-empty"),
        ([10, 5], ValueError, "must be non-empty and monotonically increasing"),
        ([[5, 3]], ValueError, "all stops must be greater than or equal"),
        ([[1, 2, 3]], ValueError, "all stops must be greater than or equal"),
        ([[[0, 1]]], TypeError, "or a list of pairs of integer starts and stops"),
    ],
)
def test_regularize_steps_errors(steps, error, message):
    with pytest.raises(error, match=message):
        uproot._util.regularize_steps(steps)


@pytest.fixture
def root_files(tmp_path):
    paths = {}
    for name in ("a.root", "b.root", "c.root"):
        paths[name] = tmp_path / name
        paths[name].touch()
    return tmp_path, paths


def regularized(files, steps_allowed=False):
    return uproot._util.regularize_files(files, steps_allowed)


def regularized_paths(files):
    # glob may join the path of its matches with the OS's separator
    return [(pathlib.Path(x), y) for x, y in regularized(files)]


def test_regularize_files_expands_braces(root_files):
    directory, paths = root_files
    assert regularized_paths(f"{directory}/{{a,c}}.root:tree") == [
        (paths["a.root"], "tree"),
        (paths["c.root"], "tree"),
    ]


def test_regularize_files_expands_several_braces(root_files):
    directory, paths = root_files
    os.mkdir(directory / "x")
    (directory / "x" / "b.root").touch()
    assert regularized_paths(f"{directory}/{{x/b,c}}{{.root,.txt}}") == [
        (directory / "x" / "b.root", None),
        (paths["c.root"], None),
    ]


def test_regularize_files_repeated_match_in_one_pattern(root_files):
    directory, paths = root_files
    # every item globs, so they all come back with the same separators
    assert sorted(regularized_paths(f"{directory}/{{a*,?}}.root:tree")) == [
        (paths["a.root"], "tree"),
        (paths["b.root"], "tree"),
        (paths["c.root"], "tree"),
    ]


def test_regularize_files_dict(root_files):
    directory, paths = root_files
    path = str(paths["a.root"])
    assert regularized({path: "tree", paths["b.root"]: None}) == [
        (path, "tree"),
        (str(paths["b.root"]), None),
    ]


def test_regularize_files_dict_same_file_as_str_and_path(root_files):
    _, paths = root_files
    path = paths["a.root"]
    assert regularized({str(path): "tree", path: "tree"}) == [(str(path), "tree")]


def test_regularize_files_dict_object_path_in_dict(root_files):
    _, paths = root_files
    path = str(paths["a.root"])
    assert regularized({path: {"object_path": "tree"}}) == [(path, "tree")]


def test_regularize_files_dict_with_steps(root_files):
    _, paths = root_files
    path = str(paths["a.root"])
    files = {path: {"object_path": "tree", "steps": [0, 10]}}
    assert regularized(files, steps_allowed=True) == [(path, "tree", [[0, 10]])]
    with pytest.raises(TypeError, match="'steps' are only allowed in uproot.dask"):
        regularized(files, steps_allowed=False)


def test_regularize_files_open_tree_has_no_steps():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root")) as file:
        tree = file["events"]
        assert regularized([tree]) == [(tree, None)]
        # an open tree is yielded as is, without a steps entry, even if allowed
        assert regularized(tree, steps_allowed=True) == [(tree, None)]


def test_regularize_files_bad_object_path():
    with pytest.raises(TypeError, match="object_path may only be a string, dict"):
        regularized({"a.root": 3})


def test_regularize_files_bad_type():
    with pytest.raises(TypeError, match="'files' must be a file path/URL.*not 3"):
        regularized(3)


def test_pandas_without_is_any_real_numeric_dtype():
    old_pandas = types.SimpleNamespace(
        api=types.SimpleNamespace(types=types.SimpleNamespace())
    )
    function = uproot._util.pandas_has_attr_is_numeric(old_pandas)
    assert function(types.SimpleNamespace(is_numeric=True)) is True
    assert function(types.SimpleNamespace(is_numeric=False)) is False


def test_unset_repr():
    assert repr(uproot._util.unset) == "uproot._util.unset"


@pytest.mark.parametrize(
    "array, expected",
    [
        (numpy.array([1]), "numpy"),
        (awkward.Array([1]), "awkward"),
        (type("Array", (), {"__module__": "cupy._core"})(), "cupy"),
        ([1], "unknown"),
    ],
)
def test_get_array_library(array, expected):
    assert uproot._util.get_array_library(array) == expected


def test_regularize_path_keeps_url_of_pathlib_path():
    path = pathlib.PurePosixPath("https://example.com/file.root")
    assert uproot._util.regularize_path(path) == "https://example.com/file.root"
