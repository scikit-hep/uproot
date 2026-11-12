# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the behavior of ``TParameter<T>``.
"""

from __future__ import annotations


class TParameter_3c_boolean_3e_:
    """
    Behaviors for ``TParameter<boolean>``.
    """

    @property
    def value(self):
        return bool(self.member("fVal"))

    def __bool__(self):
        return bool(self.member("fVal"))

    def __int__(self):
        return int(self.member("fVal"))

    def __float__(self):
        return float(self.member("fVal"))


class TParameter_3c_integer_3e_:
    """
    Behaviors for ``TParameter<integer>``.
    """

    @property
    def value(self):
        return int(self.member("fVal"))

    def __bool__(self):
        return bool(self.member("fVal"))

    def __int__(self):
        return int(self.member("fVal"))

    def __index__(self):
        return int(self.member("fVal"))

    def __float__(self):
        return float(self.member("fVal"))


class TParameter_3c_floating_3e_:
    """
    Behaviors for ``TParameter<floating>``.
    """

    @property
    def value(self):
        return float(self.member("fVal"))

    def __bool__(self):
        return bool(self.member("fVal"))

    def __int__(self):
        return int(self.member("fVal"))

    def __float__(self):
        return float(self.member("fVal"))


def TParameter(specialization):
    """
    Returns a Parameter class object for a given ``specialization``.
    """
    if specialization in ("_3c_bool_3e_", "_3c_Bool_5f_t_3e_"):
        return TParameter_3c_boolean_3e_
    elif specialization in (
        "_3c_float_3e_",
        "_3c_double_3e_",
        "_3c_long_20_double_3e_",
        "_3c_Float_5f_t_3e_",
        "_3c_Double_5f_t_3e_",
        "_3c_LongDouble_5f_t_3e_",
    ):
        return TParameter_3c_floating_3e_
    else:
        return TParameter_3c_integer_3e_
