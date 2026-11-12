# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines a versionless model for ``TClonesArray``.
"""

from __future__ import annotations

from collections.abc import Sequence

import uproot.models.TObjArray


class Model_TClonesArray(uproot.model.VersionedModel, Sequence):
    """
    A versionless :doc:`uproot.model.Model` for ``TObjArray``.

    This also satisfies Python's abstract ``Sequence`` protocol.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"memberwise serialization of {type(self).__name__}\nin file {self.file.file_path}"
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

        classname_and_version = cursor.string(chunk, context)
        try:
            i = classname_and_version.index(";")
        except ValueError:
            self._item_classname = classname_and_version
            self._item_classversion = "max"
        else:
            self._item_classname = classname_and_version[:i]
            self._item_classversion = int(classname_and_version[i + 1 :])

        self._members["fClass"] = cls = file.class_named(
            self._item_classname, self._item_classversion
        )

        self._members["fSize"], self._members["fLowerBound"] = cursor.fields(
            chunk, uproot.models.TObjArray._tobjarray_format1, context
        )

        self._data = []
        for _ in range(self._members["fSize"]):
            if cursor.byte(chunk, context) != 0:
                self._data.append(
                    cls.read(chunk, cursor, context, file, self._file, self._parent)
                )

    @property
    def item_classname(self):
        return self._item_classname

    @property
    def item_classversion(self):
        return self._item_classversion

    @property
    def item_class(self):
        return self._members["fClass"]

    def __getitem__(self, where):
        return self._data[where]

    def __len__(self):
        return len(self._data)

    base_names_versions = [("TObjArray", 3)]
    member_names = []
    class_flags = {}


uproot.classes["TClonesArray"] = Model_TClonesArray
