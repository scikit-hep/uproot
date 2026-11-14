# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines utilities for adding components to the forth reader.
"""

from __future__ import annotations

import copy
import json

import numpy as np

dtype_to_struct = {
    np.dtype("bool"): "?",
    np.dtype(">u1"): "B",
    np.dtype(">u2"): "H",
    np.dtype(">u4"): "I",
    np.dtype(">u8"): "Q",
    np.dtype(">i1"): "b",
    np.dtype(">i2"): "h",
    np.dtype(">i4"): "i",
    np.dtype(">i8"): "q",
    np.dtype(">f4"): "f",
    np.dtype(">f8"): "d",
}

struct_to_dtype_name = {
    "?": "bool",
    "B": "uint8",
    "H": "uint16",
    "I": "uint32",
    "Q": "uint64",
    "b": "int8",
    "h": "int16",
    "i": "int32",
    "q": "int64",
    "f": "float32",
    "d": "float64",
}


class SpecialPathItem:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"SpecialPathItem({self.name!r})"

    def __eq__(self, other):
        return isinstance(other, SpecialPathItem) and self.name == other.name

    def __hash__(self):
        return hash((SpecialPathItem, self.name))


def add_to_path(forth_obj, context, field):
    if forth_obj is not None:
        context = dict(context)
        context["path"] = context["path"] + (field,)
    return context


def get_first_key_number(context, extra_fields=()):
    return abs(hash(context["path"] + extra_fields))


class ForthGenerator:
    def __init__(self, interp):
        self._interp = interp
        self._active_node = self._top_node = Node("TOP")
        self._stack_of_active_nodes = []
        self._final_code = []
        self._final_header = []
        self._final_init = []

    @property
    def model(self):
        return self._top_node

    @property
    def active_node(self):
        return self._active_node

    @property
    def final_code(self):
        return self._final_code

    @property
    def final_header(self):
        return self._final_header

    @property
    def final_init(self):
        return self._final_init

    def add_node(self, new_node, current_node=None, parent_node_name=None):
        if current_node is None:
            current_node = self._top_node
        if parent_node_name is None:
            parent_node_name = self._active_node.name

        if (
            parent_node_name == current_node.name
        ) and parent_node_name != new_node.name:
            for child_node in current_node.children:
                if child_node.name == new_node.name:
                    return
            current_node.children.append(new_node)
        else:
            for child_node in current_node.children:
                self.add_node(new_node, child_node, parent_node_name)

    def reset_active_node(self):
        self._active_node = self._top_node
        self._stack_of_active_nodes = []

    def set_active_node(self, model):
        self._active_node = model

    def push_active_node(self, model):
        self._stack_of_active_nodes.append(self._active_node)
        self._active_node = model

    def pop_active_node(self):
        self._active_node = self._stack_of_active_nodes.pop()

    def _debug_forth(self):
        self._interp._debug_forth(self)


def get_forth_obj(context):
    """
    Returns a the Forth Generator object if ForthGeneration is to be done, else None.
    """
    if hasattr(context.get("forth"), "gen"):
        return context["forth"].gen
    else:
        return None


class Node:
    def __init__(
        self,
        name,
        pre_code=None,
        post_code=None,
        init_code=None,
        header_code=None,
        form_details=None,
        field_name=None,
        children=None,
    ):
        self._name = name
        self._pre_code = [] if pre_code is None else pre_code
        self._post_code = [] if post_code is None else post_code
        self._init_code = [] if init_code is None else init_code
        self._header_code = [] if header_code is None else header_code
        self._form_details = {} if form_details is None else form_details
        self._field_name = field_name
        self._children = [] if children is None else children

    def __str__(self, indent="") -> str:
        children = (",\n" + indent).join(
            x.__str__(indent + "  ") for x in self._children
        )
        if len(self._children) != 0:
            children = "\n" + indent + "  " + children + "\n" + indent
        return f"""Node({self._name!r},
{indent}  pre={"".join(self._pre_code)!r},
{indent}  post={"".join(self._post_code)!r},
{indent}  init={"".join(self._init_code)!r},
{indent}  header={"".join(self._header_code)!r},
{indent}  form_details={json.dumps(self._form_details)},
{indent}  field_name={self._field_name!r},
{indent}  children=[{children}])"""

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def pre_code(self):
        return self._pre_code

    @property
    def post_code(self):
        return self._post_code

    @property
    def header_code(self):
        return self._header_code

    @property
    def init_code(self):
        return self._init_code

    @property
    def form_details(self):
        return self._form_details

    @form_details.setter
    def form_details(self, form_details):
        self._form_details = form_details

    @property
    def field_name(self):
        return self._field_name

    @field_name.setter
    def field_name(self, new_name):
        self._field_name = new_name

    @property
    def children(self):
        return self._children

    def derive_form(self):
        if self._name.endswith(":prebuilt"):
            return self._form_details
        elif self._form_details.get("class") == "NumpyArray":
            assert len(self._children) == 0
            return self._form_details

        elif self._form_details.get("class") in ("ListOffsetArray", "RegularArray"):
            out = copy.deepcopy(self._form_details)
            if (
                self._form_details.get("parameters", {}).get("__array__")
                in ("string", "bytestring")
                and self._form_details.get("content", {}).get("class") == "NumpyArray"
            ):
                return out

            elif len(self._children) == 0:
                if out.get("content", {}).get("class") != "NumpyArray":
                    out["content"] = "NULL"
                return out

            elif (
                len(self._children) == 2
                and self._form_details.get("content", {}).get("class") == "RecordArray"
                and self._form_details["content"].get("parameters", {}).get("__array__")
                == "sorted_map"
            ):
                out["content"] = dict(out["content"])
                out["content"]["fields"] = None
                out["content"]["contents"] = []
                for child in self._children:
                    out["content"]["contents"].append(child.derive_form())
                return out

            assert len(self._children) == 1
            returnme = out
            while out.get("content", {}).get("class") in (
                "ListOffsetArray",
                "RegularArray",
            ):
                out = out["content"]
            out["content"] = self._children[0].derive_form()
            return returnme

        elif (
            self._name == "TOP"
            or self._name.startswith("dispatch-by-version ")
            or self._name.startswith("wrong-instance-version ")
            or self._name.startswith("start-of-model ")
        ):
            assert len(self._children) == 1
            return self._children[0].derive_form()

        else:
            out = dict(self._form_details)
            out["class"] = "RecordArray"
            out["fields"] = []
            out["contents"] = []
            for child in self.children:
                if child.name.startswith("base-class "):
                    assert len(child.children) == 1
                    descendant = child.children[0]
                    if descendant.name.startswith("dispatch-by-version "):
                        assert len(descendant.children) == 1
                        descendant = descendant.children[0]
                    assert descendant.name.startswith("start-of-model ")
                    assert len(descendant.children) == 1
                    base_form = descendant.children[0].derive_form()
                    out["fields"].extend(base_form["fields"])
                    out["contents"].extend(base_form["contents"])
                else:
                    descendant = child
                    while descendant.field_name is None:
                        assert descendant.form_details.get("class") in (
                            "ListOffsetArray",
                            "RegularArray",
                        )
                        assert len(descendant.children) == 1
                        descendant = descendant.children[0]
                    out["fields"].append(descendant.field_name)
                    out["contents"].append(child.derive_form())
            return out
