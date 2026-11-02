# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines a versionless model for ``THashList``.
"""

from __future__ import annotations

import uproot


class Model_THashList(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``THashList``.
    """

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:  # pragma: no cover  (always False: no header read)
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._bases.append(
            uproot.models.TList.Model_TList.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

    def __repr__(self):
        if self.class_version is None:
            version = ""
        else:
            version = f" (version {self.class_version})"
        return f"<{self.classname}{version} of {len(self)} items at 0x{id(self):012x}>"

    def __getitem__(self, where):
        return self._bases[0][where]

    def __len__(self):
        return len(self._bases[0])

    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        assert len(self._bases) == 1, "Fatal error on THashList serialization."
        for x in self._bases:
            x._serialize(out, True, None, tobject_flags)


uproot.classes["THashList"] = Model_THashList
