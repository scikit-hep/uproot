# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines a versionless model for ``TNamed``.
"""

from __future__ import annotations

import awkward
import numpy

import uproot


class Model_TNamed(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``TNamed``.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            uproot.models.TObject.Model_TObject.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._members["fName"] = cursor.string(chunk, context)
        self._members["fTitle"] = cursor.string(chunk, context)

    def __repr__(self):
        title = ""
        if self._members["fTitle"] != "":
            title = " title=" + repr(self._members["fTitle"])
        return "<TNamed {}{} at 0x{:012x}>".format(
            repr(self._members["fName"]), title, id(self)
        )

    @classmethod
    def awkward_form(cls, file, context):
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        contents["fName"] = uproot.containers.AsString(
            False, typename="TString"
        ).awkward_form(file, context)
        contents["fTitle"] = uproot.containers.AsString(
            False, typename="TString"
        ).awkward_form(file, context)
        return awkward.forms.RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TNamed"},
        )

    writable = True

    _untitled_count = 0

    def _serialize(self, out, header, name, tobject_flags):
        import uproot.writing._cascade

        where = len(out)
        self._bases[0]._serialize(
            out,
            True,
            name,
            tobject_flags | uproot.const.kIsOnHeap | uproot.const.kNotDeleted,
        )

        if name is None:
            name = self._members["fName"]
        if name is None:
            name = f"untitled_{Model_TNamed._untitled_count}"
            Model_TNamed._untitled_count += 1
        out.append(uproot.serialization.string(name))

        out.append(uproot.serialization.string(self._members["fTitle"]))
        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 1
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


uproot.classes["TNamed"] = Model_TNamed
