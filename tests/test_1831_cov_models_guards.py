# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
Hand-written models refuse what they can't read: memberwise serialization raises
NotImplementedError and AwkwardForth generation raises CannotBeForth, which
AsObjects catches to fall back on Python deserialization.
"""

from __future__ import annotations

import struct
import threading

import numpy
import pytest
import skhep_testdata

import uproot

kByteCount = 0x40000000
kMemberwise = 0x4000

memberwise_models = [
    ("TBranch", "Model_TBranch_v10"),
    ("TBranch", "Model_TBranch_v11"),
    ("TBranch", "Model_TBranch_v12"),
    ("TBranch", "Model_TBranch_v13"),
    ("TBranch", "Model_TBranchElement_v8"),
    ("TBranch", "Model_TBranchElement_v9"),
    ("TBranch", "Model_TBranchElement_v10"),
    ("TBranch", "Model_TBranchObject_v1"),
    ("TTree", "Model_TTree_v16"),
    ("TTree", "Model_TTree_v17"),
    ("TTree", "Model_TTree_v18"),
    ("TTree", "Model_TTree_v19"),
    ("TTree", "Model_TTree_v20"),
    ("TLeaf", "Model_TLeaf_v2"),
    ("TLeaf", "Model_TLeafO_v1"),
    ("TLeaf", "Model_TLeafB_v1"),
    ("TLeaf", "Model_TLeafS_v1"),
    ("TLeaf", "Model_TLeafI_v1"),
    ("TLeaf", "Model_TLeafG_v1"),
    ("TLeaf", "Model_TLeafL_v1"),
    ("TLeaf", "Model_TLeafF_v1"),
    ("TLeaf", "Model_TLeafD_v1"),
    ("TLeaf", "Model_TLeafC_v1"),
    ("TLeaf", "Model_TLeafF16_v1"),
    ("TLeaf", "Model_TLeafD32_v1"),
    ("TLeaf", "Model_TLeafElement_v1"),
    ("TTime", "Model_TTime_v2"),
    ("TMatrixT", "Model_TMatrixTSym_3c_double_3e__v5"),
    ("TTable", "Model_TDataSet"),
    ("TTable", "Model_TTableDescriptor_v4"),
    ("TTable", "Model_TTable_v4"),
    ("TNamed", "Model_TNamed"),
    ("TList", "Model_TList"),
    ("TObjArray", "Model_TObjArray"),
    ("TObjArray", "Model_TObjArrayOfTBaskets"),
    ("TObjString", "Model_TObjString"),
    ("TClonesArray", "Model_TClonesArray"),
    ("TRef", "Model_TRefArray"),
    ("TTree", "Model_ROOT_3a3a_TIOFeatures"),
    ("TTree", "Model_TTree_NumEntries"),
]

# These generate AwkwardForth instead of raising, or (TTime) are tested on real data.
not_forth_guarded = {"Model_TTime_v2", "Model_TRefArray"}

# These have no byte count and version header, so the check comes first.
forth_only_models = [
    ("TArray", "Model_TArrayC"),
    ("TArray", "Model_TArrayS"),
    ("TArray", "Model_TArrayI"),
    ("TArray", "Model_TArrayL"),
    ("TArray", "Model_TArrayL64"),
    ("TArray", "Model_TArrayF"),
    ("TArray", "Model_TArrayD"),
    ("THashList", "Model_THashList"),
    ("TBasket", "Model_TBasket"),
]


def model_class(module, name):
    return getattr(getattr(uproot.models, module), name)


def header(cls, memberwise):
    version = uproot.model.classname_version(cls.__name__) or 0
    return struct.pack(
        ">IH",
        kByteCount | 2,
        version | (kMemberwise if memberwise else 0),
    )


@pytest.fixture(scope="module")
def file():
    with uproot.open(skhep_testdata.data_path("uproot-small-flat-tree.root")) as f:
        yield f.file


def read(cls, data, file, context):
    chunk = uproot.source.chunk.Chunk.wrap(
        file.source, numpy.frombuffer(data + b"\x00" * 64, numpy.uint8)
    )
    return cls.read(
        chunk, uproot.source.cursor.Cursor(0), context, file, file.detached, None
    )


def forth_context():
    forth = threading.local()
    forth.gen = uproot._awkwardforth.ForthGenerator(None)
    return {"forth": forth, "path": ("branch",)}


@pytest.mark.parametrize("module, name", memberwise_models)
def test_memberwise(module, name, file):
    cls = model_class(module, name)
    with pytest.raises(NotImplementedError) as err:
        read(cls, header(cls, memberwise=True), file, {})
    assert str(err.value) == (
        f"memberwise serialization of {name}\nin file {file.file_path}"
    )


@pytest.mark.parametrize(
    "module, name",
    [x for x in memberwise_models if x[1] not in not_forth_guarded] + forth_only_models,
)
def test_cannot_be_forth(module, name, file):
    cls = model_class(module, name)
    with pytest.raises(uproot.interpretation.objects.CannotBeForth):
        read(cls, header(cls, memberwise=False), file, forth_context())
