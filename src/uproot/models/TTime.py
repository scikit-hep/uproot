# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module versioned model of ``TTime``.
"""

from __future__ import annotations

import struct

import numpy

import uproot


class Model_TTime_v2(uproot.model.VersionedModel):
    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                f"memberwise serialization of {type(self).__name__}\nin file {self.file.file_path}"
            )

        forth_obj = uproot._awkwardforth.get_forth_obj(context)
        if forth_obj is not None:
            key = uproot._awkwardforth.get_first_key_number(context)
            forth_stash = uproot._awkwardforth.Node(
                f"node{key} TTime :prebuilt",
                form_details={
                    "class": "RecordArray",
                    "contents": {
                        "fMilliSec": {
                            "class": "NumpyArray",
                            "primitive": "int64",
                            "form_key": f"node{key}",
                        }
                    },
                    "parameters": {"__record__": "TTime"},
                },
            )
            forth_stash.header_code.append(f"output node{key}-data int64\n")
            forth_stash.pre_code.append(f"stream !q-> node{key}-data\n")
            forth_obj.add_node(forth_stash)
            forth_obj.set_active_node(forth_stash)
        self._members["fMilliSec"] = cursor.field(chunk, self._format0, context)

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._members["fMilliSec"] = cursor.field(
                chunk, self._format_memberwise0, context
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = []
        members.append(("@num_bytes", numpy.dtype(">u4")))
        members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fMilliSec", numpy.dtype(">i8")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        contents["fMilliSec"] = uproot._util.awkward_form(
            numpy.dtype(">i8"), file, context
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TTime"},
        )

    _format0 = struct.Struct(">q")
    _format_memberwise0 = struct.Struct(">q")
    base_names_versions = []
    member_names = ["fMilliSec"]
    class_flags = {}


uproot.classes["TTime"] = Model_TTime_v2
