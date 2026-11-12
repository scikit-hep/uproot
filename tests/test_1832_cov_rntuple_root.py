# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

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
#include <string>
#include <variant>
#include <vector>

namespace test_1832 {

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

} // namespace test_1832
"""


@pytest.fixture(scope="module")
def root():
    ROOT = pytest.importorskip("ROOT")
    if ROOT.gROOT.GetVersionInt() < 63800:
        pytest.skip("needs the RNTuple interfaces of a recent ROOT")
    # ROOT warnings become Python warnings, which are errors in this test suite
    ignore_level = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kError
    try:
        if not hasattr(ROOT, "test_1832"):
            assert ROOT.gInterpreter.Declare(_WRITERS)
        yield ROOT.test_1832
    finally:
        ROOT.gErrorIgnoreLevel = ignore_level


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


def test_root_reads_packed_unions(root, tmp_path):
    path = tmp_path / "file.root"
    # a union that only uses the beginning of its contents
    data = ak.Array([1, "a", 2, "b"])[:2]
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", {"a": data})
    assert [str(root.show_variant(str(path), i)) for i in range(2)] == ["1", "a"]
