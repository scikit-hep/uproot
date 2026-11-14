# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module is initially empty, a repository for dynamically adding new classes.

The purpose of this namespace is to allow :doc:`uproot.model.VersionedModel`
classes that were automatically generated from ROOT ``TStreamerInfo`` to be
pickled, with the help of :doc:`uproot.model.DynamicModel`.

In `Python 3.7 and later <https://www.python.org/dev/peps/pep-0562>`__, attempts
to extract items from this namespace generate new :doc:`uproot.model.DynamicModel`
classes, which are used as a container in which data from pickled
:doc:`uproot.model.VersionedModel` instances are filled.
"""

from __future__ import annotations


def __getattr__(name):
    import uproot

    g = globals()
    if name not in g:
        g[name] = uproot._util.new_class(name, (uproot.model.DynamicModel,), {})

    return g[name]
