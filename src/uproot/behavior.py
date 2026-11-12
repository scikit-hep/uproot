# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines utilities for adding behaviors to C++ objects in Python.
Behaviors are defined by specially named classes in specially named modules
that get auto-detected by :doc:`uproot.behavior.behavior_of`.
"""

from __future__ import annotations

import importlib
import pkgutil

import uproot.behaviors


def behavior_of(classname):
    """
    Finds and loads the behavior class for C++ (decoded) classname or returns
    None if there isn't one.

    Behaviors do not have a required base class, and they may be used with
    Awkward Array's ``ak.behavior``.

    The search strategy for finding behavior classes is:

    1. Translate the ROOT class name from C++ to Python with
       :doc:`uproot.model.classname_encode`. For example,
       ``"ROOT::RThing"`` becomes ``"Model_ROOT_3a3a_RThing"``.
    2. Look for a submodule of ``uproot.behaviors`` without
       the ``"Model_"`` prefix. For example, ``"ROOT_3a3a_RThing"``.
    3. Look for a class in that submodule with the fully encoded
       name. For example, ``"Model_ROOT_3a3a_RThing"``.

    See :doc:`uproot.behaviors` for details.
    """
    name = uproot.model.classname_encode(classname)
    assert name.startswith("Model_")
    name = name[6:]

    specialization = None
    for param in behavior_of._specializations:
        if name.endswith(param):
            specialization = param
            name = name[: -len(param)]
            break

    if name not in globals() and name in behavior_of._module_names:
        module = importlib.import_module(f"uproot.behaviors.{name}")
        behavior_cls = getattr(module, name, None)
        if behavior_cls is not None:
            globals()[name] = behavior_cls

    cls = globals().get(name)

    if cls is None or specialization is None:
        return cls
    else:
        return cls(specialization)


behavior_of._module_names = [
    module_name
    for loader, module_name, is_pkg in pkgutil.walk_packages(uproot.behaviors.__path__)
]

behavior_of._specializations = [
    "_3c_bool_3e_",
    "_3c_char_3e_",
    "_3c_unsigned_20_char_3e_",
    "_3c_short_3e_",
    "_3c_unsigned_20_short_3e_",
    "_3c_int_3e_",
    "_3c_unsigned_20_int_3e_",
    "_3c_long_3e_",
    "_3c_unsigned_20_long_3e_",
    "_3c_long_20_long_3e_",
    "_3c_unsigned_20_long_20_long_3e_",
    "_3c_size_5f_t_3e_",
    "_3c_ssize_5f_t_3e_",
    "_3c_float_3e_",
    "_3c_double_3e_",
    "_3c_long_20_double_3e_",
    "_3c_Bool_5f_t_3e_",
    "_3c_Char_5f_t_3e_",
    "_3c_UChar_5f_t_3e_",
    "_3c_Short_5f_t_3e_",
    "_3c_UShort_5f_t_3e_",
    "_3c_Int_5f_t_3e_",
    "_3c_UInt_5f_t_3e_",
    "_3c_Long_5f_t_3e_",
    "_3c_ULong_5f_t_3e_",
    "_3c_Long64_5f_t_3e_",
    "_3c_ULong64_5f_t_3e_",
    "_3c_Size_5f_t_3e_",
    "_3c_Float_5f_t_3e_",
    "_3c_Double_5f_t_3e_",
    "_3c_LongDouble_5f_t_3e_",
]
