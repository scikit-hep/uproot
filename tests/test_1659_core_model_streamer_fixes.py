# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
Regression tests for the core model/streamer/deserialization fixes in PR #1659.
"""

import re

import pytest

import uproot
import uproot.streamers


def test_has_class_named_unknown_version():
    # Previously returned True because isinstance(cls, DispatchByVersion) was
    # always False, so the version branch was never taken.
    assert uproot.has_class_named("TH1F") is True
    assert uproot.has_class_named("TH1F", 999) is False
    assert uproot.has_class_named("Does::Not::Exist") is False


def test_class_named_known():
    cls = uproot.model.class_named("TH1F")
    assert cls.__name__ == "Model_TH1F"


def test_class_named_custom_classes_assigned():
    # Previously raised NameError because `classes` was never assigned when
    # custom_classes was not None.
    with pytest.raises(ValueError) as excinfo:
        uproot.model.class_named("TH1F", custom_classes={})
    assert "custom_classes" in str(excinfo.value)


def test_class_named_where_message_default():
    with pytest.raises(ValueError) as excinfo:
        uproot.model.class_named("Does::Not::Exist")
    assert "uproot.classes" in str(excinfo.value)


def _make_basic_type_array_element(name, fType, array_length, typename):
    element = uproot.streamers.Model_TStreamerBasicType.empty()
    element._instance_version = 4
    element._members = {
        "fName": name,
        "fTitle": "",
        "fType": fType,
        "fSize": 0,
        "fArrayLength": array_length,
        "fArrayDim": 1,
        "fTypeName": typename,
    }
    return element


def test_fixed_array_dtype_index_consistent():
    # For a fixed-size-array TStreamerBasicType member, the same _dtypeN index
    # must be referenced in both read_members and read_member_n. Previously the
    # dtypes.append happened between the two emissions, so read_member_n used
    # _dtype{N+1} (off-by-one).
    element = _make_basic_type_array_element("fArray", uproot.const.kInt, 5, "int")

    class _FakeStreamerInfo:
        name = "SyntheticClass"

    read_members = []
    read_member_n = []
    dtypes = []
    element.class_code(
        _FakeStreamerInfo(),  # streamerinfo (only used for messages here)
        0,  # i
        [element],  # elements
        read_members,
        read_member_n,
        [],  # strided_interpretation
        [],  # awkward_form
        [],  # fields
        [],  # formats
        dtypes,
        [],  # formats_memberwise
        [],  # containers
        [],  # base_names_versions
        [],  # member_names
        {},  # class_flags
        [],  # count_names
    )

    rm = "\n".join(read_members)
    rmn = "\n".join(read_member_n)

    rm_indices = set(re.findall(r"cursor\.array\(chunk, \d+, self\._dtype(\d+)", rm))
    rmn_indices = set(re.findall(r"cursor\.array\(chunk, \d+, self\._dtype(\d+)", rmn))

    assert rm_indices == {"0"}
    assert rmn_indices == {"0"}
    # exactly one dtype was registered for this single array member
    assert len(dtypes) == 1


def test_count_names_not_leaked_between_class_code_calls():
    # COUNT_NAMES used to be a module-level list shared across all class_code
    # invocations; the symbol should no longer exist as a module global.
    assert not hasattr(uproot.streamers, "COUNT_NAMES")


def test_string_classname_anchored():
    # "stringWrapper" must not be treated as std::string.
    pattern = uproot.reading._string_classname
    assert pattern.match("string")
    assert pattern.match("std::string")
    assert pattern.match("std :: string")
    assert pattern.match("stringWrapper") is None
    assert pattern.match("StringThing") is None
