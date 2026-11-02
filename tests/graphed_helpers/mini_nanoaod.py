# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""A miniature NanoAOD schema as a ``form_mapping``: the smallest object that is both halves of
``uproot._dask``'s ``ImplementsFormMapping`` / ``ImplementsFormMappingInfo`` protocol and actually
RESTRUCTURES the tree — a counter branch ``nJet`` plus flat ``Jet_*`` branches become one field
``Jet: var * MiniJet[eta, pt]``.

The restructuring is the point. Under it the graph's field names (``Jet.pt``) are no longer the
file's branch names (``Jet_pt``), and a count-only analysis needs a branch (``nJet``) that appears
nowhere in the graph — neither difference is visible through ``TrivialFormMapping``, where the two
name spaces coincide.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

#: The ``nX`` + ``X_*`` slice of a real NanoAOD file that these tests map.
FILTER = ["nJet", "Jet_pt", "Jet_eta", "nMuon", "Muon_pt", "MET_pt"]

_OFFSETS = ",!offsets"


class MiniJet(ak.Array):
    """Behavior on the mapped record: ``pt2`` is computed from ``pt``, never stored."""

    @property
    def pt2(self):
        return self.pt * self.pt


BEHAVIOR = {("*", "MiniJet"): MiniJet}


def counter_branches(fields):
    """``{"Jet": "nJet"}`` for every counter branch that has ``<name>_*`` branches beside it."""
    names = set(fields)
    return {
        f[1:]: f
        for f in fields
        if f.startswith("n") and any(k.startswith(f[1:] + "_") for k in names)
    }


class MiniNanoMapping:
    """Both protocol halves in one object, as coffea's ``_map_schema_uproot`` is.

    Leaf form keys are the ``TBranch`` name (``Jet_pt``); a collection's list form key is its
    COUNTER branch (``nJet``), so the branch that carries the list lengths falls out of the form
    and needs no counter lookup on the reader side.
    """

    def __init__(self, behavior=BEHAVIOR):
        self.behavior = behavior
        self.seen_base_form = None  # the form uproot handed the mapping

    # ---- ImplementsFormMapping ---------------------------------------------------------------
    def __call__(self, base_form):
        self.seen_base_form = base_form
        fields, contents = [], []
        counters = counter_branches(base_form.fields)
        consumed = set(counters.values())
        for name, counter in counters.items():
            sub_fields, sub_contents = [], []
            for branch in base_form.fields:
                if not branch.startswith(name + "_"):
                    continue
                jagged = base_form.content(branch)
                sub_fields.append(branch[len(name) + 1 :])
                sub_contents.append(
                    jagged.content.copy(
                        form_key=branch, parameters=dict(jagged.parameters or {})
                    )
                )
                consumed.add(branch)
            fields.append(name)
            contents.append(
                ak.forms.ListOffsetForm(
                    "i64",
                    ak.forms.RecordForm(
                        sub_contents, sub_fields, parameters={"__record__": "MiniJet"}
                    ),
                    form_key=counter,
                )
            )
        for branch in base_form.fields:
            if branch not in consumed:
                fields.append(branch)
                contents.append(base_form.content(branch).copy(form_key=branch))
        return ak.forms.RecordForm(contents, fields), self

    # ---- ImplementsFormMappingInfo -----------------------------------------------------------
    @property
    def buffer_key(self):
        return self._key_formatter

    def _key_formatter(self, form_key, form, attribute):
        if attribute == "offsets":
            form_key += _OFFSETS
        return f"/{attribute}/{form_key}"

    def parse_buffer_key(self, buffer_key):
        _prefix, attribute, form_key = buffer_key.rsplit("/", maxsplit=2)
        if attribute == "offsets":
            return form_key[: -len(_OFFSETS)], attribute
        return form_key, attribute

    def keys_for_buffer_keys(self, buffer_keys):
        return frozenset(self.parse_buffer_key(key)[0] for key in buffer_keys)

    def load_buffers(
        self,
        tree,
        keys,
        start,
        stop,
        decompression_executor=None,
        interpretation_executor=None,
        options=None,
    ):
        arrays = tree.arrays(
            list(keys),
            entry_start=start,
            entry_stop=stop,
            decompression_executor=decompression_executor,
            interpretation_executor=interpretation_executor,
        )
        counters = set(counter_branches(self.seen_base_form.fields).values())
        container = {}
        for key in keys:
            column = arrays[key]
            if key in counters:
                counts = np.asarray(column, dtype=np.int64)
                offsets = np.empty(len(counts) + 1, dtype=np.int64)
                offsets[0] = 0
                np.cumsum(counts, out=offsets[1:])
                container[f"/offsets/{key}{_OFFSETS}"] = offsets
            elif column.ndim > 1:
                container[f"/data/{key}"] = ak.to_numpy(ak.flatten(column, axis=1))
            else:
                container[f"/data/{key}"] = ak.to_numpy(column)
        return container


def eager_events(where, mapping=None, n_copies=1):
    """The parity oracle: the SAME mapping's ``load_buffers`` fed straight to
    ``awkward.from_buffers``, with every branch read — no graph, no projection, no partitions.
    """
    import uproot

    mapping = MiniNanoMapping() if mapping is None else mapping
    tree = uproot.open(where)
    keys = tree.keys(recursive=True, filter_name=FILTER, ignore_duplicates=True)
    form, info = mapping(uproot._util.get_ttree_form(tree, keys, False))
    n = tree.num_entries
    container = info.load_buffers(
        tree, frozenset(keys), 0, n, None, None, {"ak_add_doc": False}
    )
    one = ak.from_buffers(
        form, n, container, behavior=info.behavior, buffer_key=info.buffer_key
    )
    return one if n_copies == 1 else ak.concatenate([one] * n_copies)
