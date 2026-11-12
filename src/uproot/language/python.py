# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines a :doc:`uproot.language.Language` for expressions passed to
:ref:`uproot.behaviors.TBranch.HasBranches.arrays` (and similar).

The :doc:`uproot.language.python.PythonLanguage` evaluates Python code. It is
the default language.
"""

from __future__ import annotations

import ast
import warnings

import numpy

import uproot


def _expression_to_node(expression, file_path, object_path):
    try:
        node = ast.parse(expression)
    except SyntaxError as err:
        raise SyntaxError(
            err.args[0] + f"\nin file {file_path}\nin object {object_path}",
            err.args[1],
        ) from err

    if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
        raise SyntaxError(
            f"expected a single expression\nin file {file_path}\nin object {object_path}"
        )

    return node


def _attribute_to_dotted_name(node):
    if isinstance(node, ast.Attribute):
        tmp = _attribute_to_dotted_name(node.value)
        if tmp is None:
            return None
        else:
            return tmp + "." + node.attr

    elif isinstance(node, ast.Name):
        return node.id

    else:
        return None


def _walk_ast_yield_symbols(node, keys, aliases, functions, getter):
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == getter
    ):
        if (
            len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            yield node.args[0].value
        else:
            raise TypeError(
                f"expected a constant string as the only argument of {getter!r}; "
                f"found {ast.dump(node)}"
            )

    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in functions
    ):
        # A name in function-call position resolves to the function, so don't
        # treat it as a branch symbol even if a branch shares the name.
        for x in node.args:
            yield from _walk_ast_yield_symbols(x, keys, aliases, functions, getter)
        for x in node.keywords:
            yield from _walk_ast_yield_symbols(x, keys, aliases, functions, getter)

    elif isinstance(node, ast.Name):
        if node.id in keys or node.id in aliases:
            yield node.id
        elif node.id in functions or node.id == getter:
            pass
        else:
            raise KeyError(node.id)

    elif isinstance(node, ast.Attribute):
        name = _attribute_to_dotted_name(node)
        if name is None:
            yield from _walk_ast_yield_symbols(
                node.value, keys, aliases, functions, getter
            )
        elif name in keys or name in aliases:
            yield name
        else:
            # implicitly means functions and getter can't have dots in their names
            raise KeyError(name)

    elif isinstance(node, ast.AST):
        for field_name in node._fields:
            x = getattr(node, field_name)
            yield from _walk_ast_yield_symbols(x, keys, aliases, functions, getter)

    elif isinstance(node, list):
        for x in node:
            yield from _walk_ast_yield_symbols(x, keys, aliases, functions, getter)

    else:
        pass


def _ast_as_branch_expression(node, keys, aliases, functions, getter):
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == getter
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node

    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in functions
    ):
        # A name in function-call position resolves to the function, even when a
        # branch happens to share the name (e.g. a branch literally named
        # "sqrt"). Warn about the shadowing so the conflict is visible.
        if node.func.id in keys or node.func.id in aliases:
            warnings.warn(
                f"{node.func.id!r} is both a branch name and a function; "
                "using the function in call position",
                uproot.exceptions.NameConflictWarning,
                stacklevel=1,
            )
        new_func = ast.parse(f"function[{node.func.id!r}]").body[0].value
        new_args = [
            _ast_as_branch_expression(x, keys, aliases, functions, getter)
            for x in node.args
        ]
        new_keywords = [
            _ast_as_branch_expression(x, keys, aliases, functions, getter)
            for x in node.keywords
        ]
        new_node = ast.Call(new_func, new_args, new_keywords)
        new_node.lineno = getattr(node, "lineno", 1)
        new_node.col_offset = getattr(node, "col_offset", 0)
        return new_node

    elif isinstance(node, ast.Name):
        if node.id in keys or node.id in aliases:
            return ast.parse(f"get({node.id!r})").body[0].value
        elif node.id in functions:
            return ast.parse(f"function[{node.id!r}]").body[0].value
        else:
            raise KeyError(node.id)

    elif isinstance(node, ast.Attribute):
        name = _attribute_to_dotted_name(node)
        if name is None:
            value = _ast_as_branch_expression(
                node.value, keys, aliases, functions, getter
            )
            new_node = ast.Attribute(value, node.attr, node.ctx)
            new_node.lineno = getattr(node, "lineno", 1)
            new_node.col_offset = getattr(node, "col_offset", 0)
            return new_node
        elif name in keys or name in aliases:
            return ast.parse(f"get({name!r})").body[0].value
        else:
            # implicitly means functions and getter can't have dots in their names
            raise KeyError(name)

    elif isinstance(node, ast.AST):
        args = []
        for field_name in node._fields:
            field_value = getattr(node, field_name)
            args.append(
                _ast_as_branch_expression(field_value, keys, aliases, functions, getter)
            )
        new_node = type(node)(*args)
        new_node.lineno = getattr(node, "lineno", 1)
        new_node.col_offset = getattr(node, "col_offset", 0)
        return new_node

    elif isinstance(node, list):
        return [
            _ast_as_branch_expression(x, keys, aliases, functions, getter) for x in node
        ]

    else:
        return node


def _expression_to_function(
    expression, keys, aliases, functions, getter, scope, file_path, object_path
):
    if expression in keys:
        return lambda: scope[getter](expression)

    else:
        node = _expression_to_node(expression, file_path, object_path)
        try:
            expr = _ast_as_branch_expression(
                node.body[0].value, keys, aliases, functions, getter
            )
        except KeyError as err:
            raise uproot.KeyInFileError(
                err.args[0],
                keys=sorted(keys) + list(aliases),
                file_path=file_path,
                object_path=object_path,
            ) from err

        function = ast.parse("lambda: None").body[0].value
        function.body = expr
        expression = ast.Expression(function)
        expression.lineno = getattr(function, "lineno", 1)
        expression.col_offset = getattr(function, "col_offset", 0)
        return eval(compile(expression, "<dynamic>", "eval"), scope)


def _vectorized_erf(complement):
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    def erf(values):
        t = 1.0 / (numpy.absolute(values) * p + 1)
        y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * numpy.exp(
            numpy.negative(numpy.square(values))
        )
        if complement:
            return 1.0 - numpy.copysign(y, values)
        else:
            return numpy.copysign(y, values)

    return erf


def _vectorized_gamma(logarithm):
    cofs = (
        76.18009173,
        -86.50532033,
        24.01409822,
        -1.231739516e0,
        0.120858003e-2,
        -0.536382e-5,
    )
    stp = 2.50662827465

    def lgamma(values):
        x = values - 1.0
        tmp = x + 5.5
        with numpy.errstate(invalid="ignore"):
            tmp = (x + 0.5) * numpy.log(tmp) - tmp
        ser = 1.0
        with numpy.errstate(divide="ignore"):
            for cof in cofs:
                x = x + 1.0
                ser = ser + cof / x
        with numpy.errstate(invalid="ignore"):
            return tmp + numpy.log(stp * ser)

    if logarithm:
        return lgamma
    else:
        return lambda values: numpy.exp(lgamma(values))


_lgamma = _vectorized_gamma(True)


class PythonLanguage(uproot.language.Language):
    """
    Args:
        functions (None or dict): Mapping from function name to function, or
            None for ``default_functions``.
        getter (str): Name of the function that extracts branches by name;
            needed for branches whose names are not valid Python symbols.
            Default is "get".

    PythonLanguage is the default :doc:`uproot.language.Language` for
    interpreting expressions passed to
    :ref:`uproot.behaviors.TBranch.HasBranches.arrays` (and similar). This
    interpretation assumes that the expressions have Python syntax and
    semantics, with math functions loaded into the namespace.

    Unlike standard Python, an expression with attributes, such as
    ``some.thing``, can be a single identifier, so that a ``TBranch`` whose
    name contains dots does not need to be loaded with ``get("some.thing")``.
    """

    default_functions = {
        "abs": numpy.absolute,
        "absolute": numpy.absolute,
        "acos": numpy.arccos,
        "arccos": numpy.arccos,
        "acosh": numpy.arccosh,
        "arccosh": numpy.arccosh,
        "asin": numpy.arcsin,
        "arcsin": numpy.arcsin,
        "asinh": numpy.arcsinh,
        "arcsinh": numpy.arcsinh,
        "atan": numpy.arctan,
        "atan2": numpy.arctan2,
        "arctan": numpy.arctan,
        "arctan2": numpy.arctan2,
        "atanh": numpy.arctanh,
        "arctanh": numpy.arctanh,
        "cbrt": numpy.cbrt,
        "ceil": numpy.ceil,
        "conj": numpy.conjugate,
        "conjugate": numpy.conjugate,
        "copysign": numpy.copysign,
        "cos": numpy.cos,
        "cosh": numpy.cosh,
        "erf": _vectorized_erf(False),
        "erfc": _vectorized_erf(True),
        "exp": numpy.exp,
        "exp2": numpy.exp2,
        "expm1": numpy.expm1,
        "fabs": numpy.fabs,
        "factorial": lambda x: numpy.round(numpy.exp(_lgamma(numpy.round(x) + 1))),
        "floor": numpy.floor,
        "fmax": numpy.fmax,
        "fmin": numpy.fmin,
        "gamma": _vectorized_gamma(False),
        "hypot": numpy.hypot,
        "imag": numpy.imag,
        "isfinite": numpy.isfinite,
        "isinf": numpy.isinf,
        "isnan": numpy.isnan,
        "lgamma": _lgamma,
        "log": numpy.log,
        "log10": numpy.log10,
        "log1p": numpy.log1p,
        "log2": numpy.log2,
        "logical_and": numpy.logical_and,
        "logical_or": numpy.logical_or,
        "neg": numpy.negative,
        "nextafter": numpy.nextafter,
        "real": numpy.real,
        "rint": numpy.rint,
        "round": numpy.round,
        "signbit": numpy.signbit,
        "sin": numpy.sin,
        "sinh": numpy.sinh,
        "sqrt": numpy.sqrt,
        "tan": numpy.tan,
        "tanh": numpy.tanh,
        "tgamma": lambda x: numpy.exp(_lgamma(x)),
        "trunc": numpy.trunc,
        "where": numpy.where,
    }

    def __init__(self, functions=None, getter="get"):
        if functions is None:
            self._functions = self.default_functions
        else:
            self._functions = dict(functions)
        self._getter = getter

    def __repr__(self):
        return "uproot.language.python.PythonLanguage()"

    def __eq__(self, other):
        return isinstance(other, PythonLanguage)

    @property
    def functions(self):
        """
        Mapping from function name to function (dict).
        """
        return self._functions

    @property
    def getter(self):
        """
        Name of the function that extracts branches by name; needed for
        branches whose names are not valid Python symbols.
        """
        return self._getter

    def getter_of(self, name):
        """
        Returns a string, an expression in which the ``getter`` is getting
        ``name`` as a quoted string.

        For example, ``"get('something')"``.
        """
        return f"{self._getter}({name!r})"

    def free_symbols(self, expression, keys, aliases, file_path, object_path):
        """
        Args:
            expression (str): The expression to analyze.
            keys (list of str): Names of branches or aliases (for aliases that
                refer to aliases).
            aliases (list of str): Names of aliases.
            file_path (str): File path for error messages.
            object_path (str): Object path for error messages.

        Finds the symbols in the expression that are in ``keys`` or ``aliases``,
        in other words, ``TBranch`` names or alias names. These expressions may
        include dots (attributes). Known ``functions`` and the ``getter`` are
        excluded.
        """
        if expression in keys:
            return [expression]

        else:
            node = _expression_to_node(expression, file_path, object_path)
            try:
                return list(
                    _walk_ast_yield_symbols(
                        node, keys, aliases, self._functions, self._getter
                    )
                )
            except KeyError as err:
                raise uproot.KeyInFileError(
                    err.args[0], file_path=file_path, object_path=object_path
                ) from err

    def compute_expressions(
        self,
        hasbranches,
        arrays,
        expression_context,
        keys,
        aliases,
        file_path,
        object_path,
    ):
        """
        Args:
            hasbranches (:doc:`uproot.behaviors.TBranch.HasBranches`): The
                ``TTree`` or ``TBranch`` that is requesting a computation.
            arrays (dict of arrays): Inputs to the computation.
            expression_context (list of (str, dict) tuples): Expression strings
                and a dict of metadata about each.
            keys (set of str): Names of branches or aliases (for aliases that
                refer to aliases).
            aliases (dict of str \u2192 str): Names of aliases and their definitions.
            file_path (str): File path for error messages.
            object_path (str): Object path for error messages.

        Computes an array for each expression.
        """
        values = {}

        if len(aliases) < len(keys):
            shorter, longer = aliases, keys
        else:
            shorter, longer = keys, aliases
        for x in shorter:
            if x in longer:
                warnings.warn(
                    f"{x!r} is both an alias and a branch name",
                    uproot.exceptions.NameConflictWarning,
                    stacklevel=1,
                )

        def getter(name):
            if name not in values:
                values[name] = _expression_to_function(
                    aliases[name],
                    keys,
                    aliases,
                    self._functions,
                    self._getter,
                    scope,
                    file_path,
                    object_path,
                )()
            return values[name]

        scope = {self._getter: getter, "function": self._functions}
        for _, context in expression_context:
            for branch in context["branches"]:
                array = arrays[branch.cache_key]
                name = branch.name
                while branch is not hasbranches:
                    if name in keys:
                        values[name] = array
                    branch = branch.parent  # noqa: PLW2901 (overwriting branch)
                    if branch is not hasbranches:
                        name = branch.name + "/" + name
                name = "/" + name
                if name in keys:
                    values[name] = array

        output = {}
        is_pandas = False
        for expression, context in expression_context:
            if context["is_primary"] and not context["is_cut"]:
                output[expression] = _expression_to_function(
                    expression,
                    keys,
                    aliases,
                    self._functions,
                    self._getter,
                    scope,
                    file_path,
                    object_path,
                )()
                if uproot._util.from_module(output[expression], "pandas"):
                    is_pandas = True

        cut = None
        for expression, context in expression_context:
            if context["is_primary"] and context["is_cut"]:
                cut = _expression_to_function(
                    expression,
                    keys,
                    aliases,
                    self._functions,
                    self._getter,
                    scope,
                    file_path,
                    object_path,
                )()
                if uproot._util.from_module(cut, "pandas"):
                    is_pandas = True
                break

        if cut is not None:
            cut = cut != 0

            if is_pandas:
                pandas = uproot.extras.pandas()

            for name, data in output.items():
                if (
                    is_pandas
                    and isinstance(cut.index, pandas.MultiIndex)
                    and not isinstance(data.index, pandas.MultiIndex)
                ):
                    original = data
                    modified = pandas.DataFrame(
                        {original.name: original.values},
                        index=pandas.MultiIndex.from_arrays(
                            [original.index], names=["entry"]
                        ),
                    ).reindex(cut.index)
                    selected = modified[cut]
                    output[name] = selected[original.name]

                else:
                    output[name] = data[cut]

        # clear dicts to get rid of big arrays.
        # note: without this these arrays are not properly released from memory!
        values.clear()
        scope.clear()
        return output


python_language = PythonLanguage()
