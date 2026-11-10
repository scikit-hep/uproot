# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot
from uproot.interpretation.identify import parse_typename


@pytest.fixture(scope="module")
def datafile(tmpdir_factory):
    yield skhep_testdata.data_path("uproot-issue-1229.root")


@pytest.fixture
def tree(datafile):
    with uproot.open(datafile) as f:
        yield f["tree"]


def test_const_in_typename(tree):
    assert tree["branch/pointer"].typename == "TFooMember*"
    assert tree["branch/const_pointer"].typename == "TFooMember*"
    for name in ["branch/pointer", "branch/const_pointer"]:
        (member,) = tree[name].array(library="np")
        assert type(member).__name__.startswith("Model_TFooMember")


def test_const_parse_typename():
    assert parse_typename("TH1I*") == parse_typename("const TH1I*")
