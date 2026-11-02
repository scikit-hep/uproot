# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import json

import numpy as np
import pytest

import uproot

ak = pytest.importorskip("awkward")

_WRITERS = """
#include <ROOT/RField.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <TFile.h>
#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace test_1814 {

struct Base {
   int member;
   float other;
};
struct Child : Base {
   int member;
};
struct Left {
   int shared;
};
struct Right {
   int shared;
};
struct Both : Left, Right {
   int member;
};

void write_projections(const char *path)
{
   auto model = ROOT::RNTupleModel::Create();
   auto pt = model->MakeField<float>("pt");
   auto values = model->MakeField<std::vector<int>>("values");
   model
      ->AddProjectedField(ROOT::RFieldBase::Create("pt_alias", "float").Unwrap(),
                          [](const std::string &) { return std::string("pt"); })
      .ThrowOnError();
   model
      ->AddProjectedField(
         ROOT::RFieldBase::Create("num_values", "ROOT::RNTupleCardinality<std::uint64_t>").Unwrap(),
         [](const std::string &) { return std::string("values"); })
      .ThrowOnError();
   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntuple", path);
   for (int i = 0; i < 4; i++) {
      *pt = 1.5f * i;
      *values = std::vector<int>(i, i);
      writer->Fill();
   }
}

void write_shadowed_members(const char *path)
{
   auto model = ROOT::RNTupleModel::Create();
   auto child = model->MakeField<Child>("child");
   auto both = model->MakeField<Both>("both");
   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntuple", path);
   for (int i = 0; i < 3; i++) {
      child->member = i;
      child->Base::member = 10 * i;
      child->other = 0.5f * i;
      both->member = 100 * i;
      both->Left::shared = 200 * i;
      both->Right::shared = 300 * i;
      writer->Fill();
   }
}

void write_streamer_field(const char *path)
{
   auto model = ROOT::RNTupleModel::Create();
   model->MakeField<int>("x");
   model->AddField(std::make_unique<ROOT::RStreamerField>("streamed", "std::vector<int>"));
   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntuple", path);
   auto entry = writer->CreateEntry();
   auto x = entry->GetPtr<int>("x");
   auto streamed = entry->GetPtr<std::vector<int>>("streamed");
   for (int i = 0; i < 2; i++) {
      *x = i;
      streamed->push_back(i);
      writer->Fill(*entry);
   }
}

void write_without_fields(const char *path)
{
   auto writer = ROOT::RNTupleWriter::Recreate(ROOT::RNTupleModel::Create(), "ntuple", path);
}

std::string show(const char *path, int entry)
{
   std::ostringstream stream;
   ROOT::RNTupleReader::Open("ntuple", path)->Show(entry, stream);
   return stream.str();
}

// Show doesn't support variants
std::string show_variant(const char *path, int entry)
{
   auto reader = ROOT::RNTupleReader::Open("ntuple", path);
   auto view = reader->GetView<std::variant<std::int64_t, std::string>>("a");
   auto value = view(entry);
   if (value.index() == 0)
      return std::to_string(std::get<0>(value));
   return std::get<1>(value);
}

} // namespace test_1814
"""

_ATTRIBUTE_WRITER = """
#include <ROOT/RNTupleAttrWriting.hxx>

namespace test_1814 {

void write_attributes(const char *path)
{
   auto file = std::unique_ptr<TFile>(TFile::Open(path, "RECREATE"));
   auto model = ROOT::RNTupleModel::Create();
   auto x = model->MakeField<int>("x");
   auto writer = ROOT::RNTupleWriter::Append(std::move(model), "ntuple", *file);
   auto attribute_model = ROOT::RNTupleModel::Create();
   auto note = attribute_model->MakeField<std::string>("note");
   auto attributes = writer->CreateAttributeSet(std::move(attribute_model), "my_attributes");
   auto range = attributes->BeginRange();
   *note = "a note";
   for (int i = 0; i < 4; i++) {
      *x = i;
      writer->Fill();
   }
   attributes->CommitRange(std::move(range));
   writer.reset();
}

} // namespace test_1814
"""


@pytest.fixture(scope="module")
def root():
    ROOT = pytest.importorskip("ROOT")
    if ROOT.gROOT.GetVersionInt() < 63800:
        pytest.skip("needs the RNTuple interfaces of a recent ROOT")
    # ROOT warnings become Python warnings, which are errors in this test suite. The
    # classes above have no dictionaries, which ROOT warns about.
    ignore_level = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kError
    if not hasattr(ROOT, "test_1814"):
        assert ROOT.gInterpreter.Declare(_WRITERS)
        if ROOT.gROOT.GetVersionInt() >= 64000:
            assert ROOT.gInterpreter.Declare(_ATTRIBUTE_WRITER)
    yield ROOT.test_1814
    ROOT.gErrorIgnoreLevel = ignore_level


def _show(root, path, entry):
    return json.loads(str(root.show(str(path), entry)))


def test_projected_fields(root, tmp_path):
    path = tmp_path / "file.root"
    root.write_projections(str(path))
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        assert ntuple.keys() == ["pt", "values", "pt_alias", "num_values"]

        alias_field = ntuple["pt_alias"].field_id
        assert ntuple.field_records[alias_field].source_field_id == 0
        assert [
            (alias.physical_id, alias.field_id) for alias in ntuple.alias_column_records
        ] == [(0, alias_field), (1, ntuple["num_values"].field_id)]
        assert ntuple.alias_column_records is ntuple.alias_column_records

        # the form of a projected field is the one of its source field
        assert ntuple.col_form(alias_field) == ntuple.col_form(0)

        out = ntuple.arrays()
        assert out.pt.to_list() == [0.0, 1.5, 3.0, 4.5]
        assert out.pt_alias.to_list() == out.pt.to_list()
        assert out.values.to_list() == [[], [1], [2, 2], [3, 3, 3]]
        assert out.num_values.to_list() == [0, 1, 2, 3]


def test_virtual_cardinality_field(root, tmp_path):
    path = tmp_path / "file.root"
    root.write_projections(str(path))
    with uproot.open(path) as file:
        out = file["ntuple"].arrays(filter_name="num_values", virtual=True)
        assert ak.to_list(out.num_values) == [0, 1, 2, 3]


def test_members_that_shadow_inherited_members(root, tmp_path):
    path = tmp_path / "file.root"
    root.write_shadowed_members(str(path))
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        assert sorted(ntuple["child"].keys()) == [
            "member",
            "other",
            "test_1814::Base::member",
        ]
        # members of different bases are told apart by their own prefix, and the one
        # that is not repeated in a base stays as it is
        assert sorted(ntuple["both"].keys()) == [
            "member",
            "test_1814::Left::shared",
            "test_1814::Right::shared",
        ]
        out = ntuple.arrays()
        assert out.child.member.to_list() == [0, 1, 2]
        assert out.child["test_1814::Base::member"].to_list() == [0, 10, 20]
        assert out.both["test_1814::Left::shared"].to_list() == [0, 200, 400]
        assert out.both["test_1814::Right::shared"].to_list() == [0, 300, 600]


def test_streamer_field(root, tmp_path):
    path = tmp_path / "file.root"
    root.write_streamer_field(str(path))
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        record = ntuple["streamed"].record
        assert record.struct_role == uproot.const.RNTupleFieldRole.STREAMER
        assert record.type_name == "std::vector<std::int32_t>"
        assert isinstance(record.checksum, int)
        assert ntuple.arrays(filter_name="x").x.to_list() == [0, 1]
        with pytest.raises(NotImplementedError, match="Unsplit fields are not"):
            ntuple.arrays()


def test_rntuple_without_fields(root, tmp_path):
    path = tmp_path / "file.root"
    root.write_without_fields(str(path))
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        assert len(ntuple) == 0
        assert ntuple.keys() == []
        assert ntuple.num_entries == 0
        assert repr(ntuple).startswith("<ROOT::RNTuple 'ntuple' at 0x")


def test_linked_attribute_set(root, tmp_path):
    if not hasattr(root, "write_attributes"):
        pytest.skip("needs RNTuple attributes")
    path = tmp_path / "file.root"
    root.write_attributes(str(path))
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        linked = ntuple.footer.linked_attribute_sets
        assert [record.attribute_set_name for record in linked] == ["my_attributes"]
        assert ntuple.arrays().x.to_list() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            ak.contents.BitMaskedArray(
                ak.index.IndexU8(np.array([0b101], dtype=np.uint8)),
                ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int32)),
                valid_when=True,
                length=3,
                lsb_order=True,
            ),
            [1, None, 3],
        ),
        (
            ak.contents.UnmaskedArray(
                ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32))
            ),
            [1, 2],
        ),
    ],
)
def test_root_reads_packed_layouts(root, tmp_path, data, expected):
    path = tmp_path / "file.root"
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", {"a": ak.Array(data)})
    assert [_show(root, path, i)["a"] for i in range(len(expected))] == expected


def test_root_reads_packed_unions(root, tmp_path):
    path = tmp_path / "file.root"
    # a union that only uses the beginning of its contents
    data = ak.Array([1, "a", 2, "b"])[:2]
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", {"a": data})
    assert [str(root.show_variant(str(path), i)) for i in range(2)] == ["1", "a"]
