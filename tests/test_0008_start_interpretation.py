# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import sys

import numpy
import pytest
import skhep_testdata

import uproot


def tobytes(x):
    if hasattr(x, "tobytes"):
        return x.tobytes()
    else:
        return x.tostring()


def test_classname_encoding(tmpdir):
    assert (
        uproot.model.classname_encode("namespace::some.deep<templated, thing>", 12)
        == "Model_namespace_3a3a_some_2e_deep_3c_templated_2c20_thing_3e__v12"
    )
    assert (
        uproot.model.classname_encode("namespace::some.deep<templated, thing>")
        == "Model_namespace_3a3a_some_2e_deep_3c_templated_2c20_thing_3e_"
    )

    assert uproot.model.classname_decode(
        "Model_namespace_3a3a_some_2e_deep_3c_templated_2c20_thing_3e__v12"
    ) == ("namespace::some.deep<templated, thing>", 12)
    assert uproot.model.classname_decode(
        "Model_namespace_3a3a_some_2e_deep_3c_templated_2c20_thing_3e_"
    ) == ("namespace::some.deep<templated, thing>", None)


def test_file_header():
    filename = skhep_testdata.data_path("uproot-Zmumu.root")
    file = uproot.reading.ReadOnlyFile(filename)
    assert repr(file.compression) == "ZLIB(4)"
    assert not file.is_64bit
    assert file.fNbytesInfo == 4447
    assert file.hex_uuid == "944b77d0-98ab-11e7-a769-0100007fbeef"
