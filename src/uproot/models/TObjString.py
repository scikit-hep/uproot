# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines a versionless model for ``TObjString``.
"""

from __future__ import annotations

import awkward

import uproot
import uproot.serialization


class Model_TObjString(uproot.model.Model, str):
    """
    A versionless :doc:`uproot.model.Model` for ``TObjString``.

    This is also a Python ``str`` (string).
    """

    class_rawstreamers = (
        (
            None,
            b"@\x00\x01X\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01B\x00\t@\x00\x00\x18\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\nTObjString\x00\x9c\x8eH\x00\x00\x00\x00\x01@\x00\x01\x18\xff\xff\xff\xffTObjArray\x00@\x00\x01\x06\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00u\xff\xff\xff\xffTStreamerBase\x00@\x00\x00_\x00\x03@\x00\x00U\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TObject\x11Basic ROOT object\x00\x00\x00B\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x90\x1b\xc0-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00t\xff\xff\xff\xffTStreamerString\x00@\x00\x00\\\x00\x02@\x00\x00V\x00\x04@\x00\x00$\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fString\x0fwrapped TString\x00\x00\x00A\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TString\x00",
            "TObjString",
            1,
        ),
    )
    writable = True

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
        self._data = cursor.string(chunk, context)
        # ROOT's name for it, which models generated from StreamerInfo also use
        self._members["fString"] = self._data

    def postprocess(self, chunk, cursor, context, file):
        out = Model_TObjString(self._data)
        out._cursor = self._cursor
        out._file = self._file
        out._parent = self._parent
        out._members = self._members
        out._bases = self._bases
        out._num_bytes = self._num_bytes
        out._instance_version = self._instance_version
        out._is_memberwise = self._is_memberwise
        return out

    def _to_writable(self, concrete):
        # Model._to_writable would make an empty str and fill in its members,
        # but TObjString has only one version, so it's writable as it is.
        return self

    @property
    def fTitle(self):
        return "Collectable string class"

    def tojson(self):
        out = self._bases[0].tojson()  # TObject
        out["_typename"] = self.classname
        out["fString"] = str(self)
        return out

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, name, tobject_flags | uproot.const.kNotDeleted)
        out.append(uproot.serialization.string(str(self)))
        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 1
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))

    def __repr__(self):
        if self.class_version is None:
            version = ""
        else:
            version = f" (version {self.class_version})"
        return f"<{self.classname}{version} {str.__repr__(self)} at 0x{id(self):012x}>"

    @classmethod
    def awkward_form(cls, file, context):
        return awkward.forms.ListOffsetForm(
            context["index_format"],
            awkward.forms.NumpyForm("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        )


uproot.classes["TObjString"] = Model_TObjString
