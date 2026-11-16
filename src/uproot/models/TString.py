# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines a versionless model of ``TString``.
"""

from __future__ import annotations

import uproot
import uproot._awkwardforth


class Model_TString(uproot.model.Model, str):
    """
    A versionless :doc:`uproot.model.Model` for ``TString``.

    This is also a Python ``str`` (string).
    """

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        forth_obj = uproot._awkwardforth.get_forth_obj(context)
        if forth_obj is not None:
            offsets_num = uproot._awkwardforth.get_first_key_number(context)
            data_num = offsets_num + 1
            nested_forth_stash = uproot._awkwardforth.Node(
                f"node{offsets_num} TString",
                form_details={
                    "class": "ListOffsetArray",
                    "offsets": "i64",
                    "content": {
                        "class": "NumpyArray",
                        "primitive": "uint8",
                        "inner_shape": [],
                        "parameters": {"__array__": "char"},
                        "form_key": f"node{data_num}",
                    },
                    "parameters": {"__array__": "string"},
                    "form_key": f"node{offsets_num}",
                },
            )
            nested_forth_stash.pre_code.append(
                f" stream !B-> stack dup 255 = if drop stream !I-> stack then dup node{offsets_num}-offsets +<- stack stream #!B-> node{data_num}-data\n"
            )
            nested_forth_stash.header_code.append(
                f"output node{offsets_num}-offsets int64\noutput node{data_num}-data uint8\n"
            )
            nested_forth_stash.init_code.append(
                f"0 node{offsets_num}-offsets <- stack\n"
            )
            forth_obj.add_node(nested_forth_stash)

        self._data = cursor.string(chunk, context)

    def postprocess(self, chunk, cursor, context, file):
        out = Model_TString(self._data)
        out._cursor = self._cursor
        out._file = self._file
        out._parent = self._parent
        out._members = self._members
        out._bases = self._bases
        out._num_bytes = self._num_bytes
        out._instance_version = self._instance_version
        return out

    def __repr__(self):
        if self.class_version is None:
            version = ""
        else:
            version = f" (version {self.class_version})"
        return f"<{self.classname}{version} {str.__repr__(self)} at 0x{id(self):012x}>"

    def tojson(self):
        return str(self)

    @classmethod
    def awkward_form(cls, file, context):
        return uproot.containers.AsString(False, typename="TString").awkward_form(
            file, context
        )

    writable = True
    _is_memberwise = False

    def _serialize(self, out, header, name, tobject_flags):
        import uproot.writing._cascade

        where = len(out)
        for x in self._bases:
            x._serialize(out, True, None, tobject_flags)

        out.append(uproot.serialization.string(self))

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 2
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


uproot.classes["TString"] = Model_TString
