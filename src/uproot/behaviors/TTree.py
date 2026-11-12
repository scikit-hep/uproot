# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the behaviors of ``TTree``, which is almost entirely inherited from
functions in :doc:`uproot.behaviors.TBranch`.
"""

from __future__ import annotations

import uproot


class TTree(uproot.behaviors.TBranch.HasBranches):
    """
    Behaviors for a ``TTree``, which mostly consist of array-reading methods.

    Since a :doc:`uproot.behaviors.TTree.TTree` is a
    :doc:`uproot.behaviors.TBranch.HasBranches`, it is also a Python
    ``Mapping``, which uses square bracket syntax to extract subbranches:

    .. code-block:: python

        my_tree["branch"]
        my_tree["branch"]["subbranch"]
        my_tree["branch/subbranch"]
        my_tree["branch/subbranch/subsubbranch"]
    """

    def __repr__(self):
        if len(self) == 0:
            return f"<TTree {self.name!r} at 0x{id(self):012x}>"
        else:
            return f"<TTree {self.name!r} ({len(self)} branches) at 0x{id(self):012x}>"

    @property
    def name(self):
        """
        Name of the ``TTree``.
        """
        return self.member("fName")

    @property
    def title(self):
        """
        Title of the ``TTree``.
        """
        return self.member("fTitle")

    @property
    def object_path(self):
        """
        Object path of the ``TTree``.
        """
        return self.parent.object_path

    @property
    def cache_key(self):
        """
        String that uniquely specifies this ``TTree`` in its path, to use as
        part of object and array cache keys.
        """
        return f"{self.parent.parent.cache_key}{self.name};{self.parent.fCycle}"

    @property
    def num_entries(self):
        """
        The number of entries in the ``TTree``, as reported by ``fEntries``.

        In principle, this could disagree with the
        :ref:`uproot.behaviors.TBranch.TBranch.num_entries`, which is from the
        ``TBranch``'s ``fEntries``. See that method's documentation for yet more
        ways the number of entries could, in principle, be inconsistent.
        """
        return self.member("fEntries")

    @property
    def tree(self):
        """
        Returns ``self`` because this is a ``TTree``.
        """
        return self

    @property
    def compressed_bytes(self):
        """
        The number of compressed bytes in all ``TBaskets`` of all ``TBranches``
        of this ``TTree``, including all the TKey headers (which are always
        uncompressed).

        This information is specified in the ``TTree`` metadata (``fZipBytes``)
        and can be determined without reading any additional data.
        """
        return self.member("fZipBytes")

    @property
    def uncompressed_bytes(self):
        """
        The number of uncompressed bytes in all ``TBaskets`` of all ``TBranches``
        of this ``TTree``, including all the TKey headers.

        This information is specified in the ``TTree`` metadata (``fTotBytes``)
        and can be determined without reading any additional data.
        """
        return self.member("fTotBytes")

    @property
    def compression_ratio(self):
        """
        The number of uncompressed bytes divided by the number of compressed
        bytes for this ``TTree``.

        See :ref:`uproot.behaviors.TTree.TTree.compressed_bytes` and
        :ref:`uproot.behaviors.TTree.TTree.uncompressed_bytes`.
        """
        return float(self.uncompressed_bytes) / float(self.compressed_bytes)

    @property
    def aliases(self):
        """
        The ``TTree``'s ``fAliases``, which are used as the ``aliases``
        argument to :ref:`uproot.behaviors.TBranch.HasBranches.arrays`,
        :ref:`uproot.behaviors.TBranch.HasBranches.iterate`,
        :doc:`uproot.behaviors.TBranch.iterate`, and
        :doc:`uproot.behaviors.TBranch.concatenate` if one is not given.

        The return type is always a dict of str \u2192 str, even if there
        are no aliases (an empty dict).
        """
        aliases = self.member("fAliases", none_if_missing=True)
        if aliases is None:
            return {}
        else:
            return {alias.member("fName"): alias.member("fTitle") for alias in aliases}

    @property
    def chunk(self):
        """
        The :doc:`uproot.source.chunk.Chunk` from which this ``TTree`` was
        read (as a strong, not weak, reference).

        The reason the chunk is retained is to read
        :ref:`uproot.behaviors.TBranch.TBranch.embedded_baskets` only if
        necessary (if the file was opened with
        ``options["minimal_ttree_metadata"]=True``, the reading of these
        ``TBaskets`` is deferred until they are accessed).

        Holding a strong reference to a chunk holds a strong reference to
        its :doc:`uproot.source.chunk.Source`, preventing open file handles
        from going out of scope, but so does the
        :doc:`uproot.reading.ReadOnlyFile` that ``TTree`` needs to read data
        on demand.
        """
        return self._chunk

    def postprocess(self, chunk, cursor, context, file):
        self._chunk = chunk
        self._lookup = {}
        for branch in self.member("fBranches"):
            name = branch.member("fName")
            if name not in self._lookup:
                self._lookup[name] = branch
        return self
