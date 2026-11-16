# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Tests of uproot.language.python, the default language of expressions.
"""

from __future__ import annotations

import math

import numpy
import pytest

import uproot
from uproot.language.python import _expression_to_function, PythonLanguage


@pytest.fixture
def tree(tmp_path):
    path = tmp_path / "language.root"
    with uproot.recreate(path) as file:
        file.mktree("tree", {"x": "float64", "y": "float64"})
        file["tree"].extend({"x": [1.0, 4.0, 9.0], "y": [-1.0, 0.25, 2.5]})

    with uproot.open(path) as file:
        yield file["tree"]


def compute(tree, expression, **kwargs):
    return tree.arrays(expression, library="np", **kwargs)[expression]


@pytest.mark.parametrize("expression", ["z = x", "x; y", "import x", ""])
def test_not_a_single_expression(tree, expression):
    with pytest.raises(SyntaxError) as excinfo:
        tree.arrays([expression], library="np")
    assert excinfo.value.msg == (
        "expected a single expression\n"
        f"in file {tree.file.file_path}\nin object {tree.object_path}"
    )


def test_syntax_error(tree):
    with pytest.raises(SyntaxError) as excinfo:
        tree.arrays(["x +"], library="np")
    assert f"in file {tree.file.file_path}\nin object" in excinfo.value.msg


def test_attribute_of_a_computed_value(tree):
    assert compute(tree, "(x + y).real").tolist() == [0.0, 4.25, 11.5]


def test_function_keywords(tree):
    assert compute(tree, "round(x / 3, decimals=1)").tolist() == [0.3, 1.3, 3.0]


def test_getter(tree):
    assert compute(tree, "get('x') * 2").tolist() == [2.0, 8.0, 18.0]


def test_function_as_a_value(tree):
    language = PythonLanguage(
        functions={"apply": lambda function, x: function(x), "sqrt": numpy.sqrt}
    )
    assert compute(tree, "apply(sqrt, x)", language=language).tolist() == [
        1.0,
        2.0,
        3.0,
    ]


def test_getter_as_a_value_is_not_a_symbol(tree):
    language = PythonLanguage(functions={"apply": lambda function, x: function(x)})

    with pytest.raises(uproot.KeyInFileError) as excinfo:
        compute(tree, "apply(get, x)", language=language)
    assert excinfo.value.key == "get"
    assert excinfo.value.keys == ["/x", "/y", "x", "y"]
    assert excinfo.value.file_path == tree.file.file_path


@pytest.mark.parametrize(
    "expression, key",
    [("x + nothing", "nothing"), ("nothing.at.all", "nothing.at.all")],
)
def test_unknown_symbols(tree, expression, key):
    with pytest.raises(uproot.KeyInFileError) as excinfo:
        tree.arrays([expression], library="np")
    assert excinfo.value.key == key


@pytest.mark.parametrize(
    "expression", ["some.thing", "(x + 1).thing.other + some.thing"]
)
def test_expression_to_function_with_unknown_attributes(expression):
    with pytest.raises(uproot.KeyInFileError) as excinfo:
        _expression_to_function(
            expression, {"x"}, {}, {}, "get", {}, "file.root", "/tree;1"
        )
    assert excinfo.value.key == "some.thing"
    assert excinfo.value.file_path == "file.root"
    assert excinfo.value.object_path == "/tree;1"


def test_expression_to_function_with_unknown_name():
    with pytest.raises(uproot.KeyInFileError) as excinfo:
        _expression_to_function("z", {"x"}, {"a": "x"}, {}, "get", {}, "f", "o")
    assert excinfo.value.key == "z"
    assert excinfo.value.keys == ["x", "a"]


def test_expression_to_function_with_dotted_names():
    scope = {"get": {"some.thing": 5}.__getitem__}
    function = _expression_to_function(
        "some.thing + 1", {"some.thing"}, {}, {}, "get", scope, "f", "o"
    )
    assert function() == 6


@pytest.mark.parametrize("value", [-2.5, -0.5, 0.0, 0.1, 1.0, 3.0])
def test_erf(value):
    functions = PythonLanguage.default_functions
    assert functions["erf"](numpy.array([value]))[0] == pytest.approx(
        math.erf(value), abs=1e-6
    )
    assert functions["erfc"](numpy.array([value]))[0] == pytest.approx(
        math.erfc(value), abs=1e-6
    )


@pytest.mark.parametrize("value", [0.5, 1.0, 2.0, 3.5, 10.0])
def test_gamma(value):
    functions = PythonLanguage.default_functions
    array = numpy.array([value])

    assert functions["lgamma"](array)[0] == pytest.approx(
        math.lgamma(value), rel=1e-6, abs=1e-6
    )
    assert functions["gamma"](array)[0] == pytest.approx(math.gamma(value), rel=1e-6)
    assert functions["tgamma"](array)[0] == pytest.approx(math.gamma(value), rel=1e-6)


def test_factorial():
    factorial = PythonLanguage.default_functions["factorial"]
    assert factorial(numpy.array([0.0, 1.0, 5.0, 10.0])).tolist() == [
        1.0,
        1.0,
        120.0,
        3628800.0,
    ]


def test_math_in_expressions(tree):
    assert compute(tree, "factorial(sqrt(x))").tolist() == [1.0, 2.0, 6.0]
    assert compute(tree, "erfc(y)")[0] == pytest.approx(math.erfc(-1.0), abs=1e-6)


def test_language_properties():
    default = PythonLanguage()
    assert repr(default) == "uproot.language.python.PythonLanguage()"
    assert default == uproot.language.python.python_language
    assert default != 5
    assert default.functions is PythonLanguage.default_functions
    assert default.getter == "get"
    assert default.getter_of("some/name") == "get('some/name')"

    custom = PythonLanguage(functions={"f": abs}, getter="fetch")
    assert custom.functions == {"f": abs}
    assert custom.getter == "fetch"
    assert custom.getter_of("x") == "fetch('x')"


def test_custom_getter(tree):
    language = PythonLanguage(getter="fetch")
    assert compute(tree, "fetch('x') + 1", language=language).tolist() == [
        2.0,
        5.0,
        10.0,
    ]


def primary(expression, branches):
    return (
        expression,
        {"is_primary": True, "is_cut": False, "branches": branches},
    )


def cut(expression, branches):
    return (
        expression,
        {"is_primary": True, "is_cut": True, "branches": branches},
    )


def test_compute_expressions_ignores_unrequested_paths(tree):
    # "/x", the name with respect to the top of the tree, was not requested
    x = tree["x"]
    array = numpy.array([1.0, 2.0])

    output = PythonLanguage().compute_expressions(
        tree, {x.cache_key: array}, [primary("x", [x])], {"x"}, {}, "f", "o"
    )

    assert list(output) == ["x"]
    assert output["x"] is array


def test_compute_expressions_with_absolute_path(tree):
    x = tree["x"]
    array = numpy.array([1.0, 2.0])

    output = PythonLanguage().compute_expressions(
        tree, {x.cache_key: array}, [primary("/x", [x])], {"/x"}, {}, "f", "o"
    )

    assert output["/x"] is array


@pytest.mark.parametrize(
    "aliases", [{"x": "y"}, {"x": "y", "z": "x", "w": "x", "v": "x"}]
)
def test_alias_with_the_name_of_a_branch(tree, aliases):
    x = tree["x"]
    array = numpy.array([1.0, 2.0])

    with pytest.warns(
        uproot.exceptions.NameConflictWarning,
        match=r"^'x' is both an alias and a branch name$",
    ):
        output = PythonLanguage().compute_expressions(
            tree,
            {x.cache_key: array},
            [primary("x", [x])],
            {"x", "y"},
            aliases,
            "f",
            "o",
        )

    assert output["x"] is array


def test_cut_with_a_multiindex(tree):
    pandas = pytest.importorskip("pandas")

    x, y = tree["x"], tree["y"]
    flat = pandas.Series([1.0, 2.0], name="x")
    index = pandas.MultiIndex.from_arrays(
        [[0, 0, 1], [0, 1, 0]], names=["entry", "subentry"]
    )
    selection = pandas.Series([1.0, 0.0, 1.0], index=index)

    output = PythonLanguage().compute_expressions(
        tree,
        {x.cache_key: flat, y.cache_key: selection},
        [primary("x", [x]), cut("y", [y])],
        {"x", "y"},
        {},
        "f",
        "o",
    )

    assert output["x"].index.tolist() == [(0, 0), (1, 0)]
    # entries 0 and 1 pass the cut, each with its own flat x
    assert output["x"].tolist() == [1.0, 2.0]
