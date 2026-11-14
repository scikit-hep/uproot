# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
"""TBasket read counting and the basket-alignment rule the aligned-partition tests use.

The rule: each boundary of the unaligned split moves to the nearest entry offset shared
by every ``TBranch`` that has baskets (a tie goes to the lower offset), and ranges left
empty are dropped.
"""

from __future__ import annotations

import contextlib
import threading
from collections import Counter

import uproot.models.TBasket


@contextlib.contextmanager
def count_baskets():
    """Counts ``(branch, basket number)`` per full ``TBasket`` read in the block."""
    counts = Counter()
    lock = threading.Lock()
    model = uproot.models.TBasket.Model_TBasket
    original = model.read_members

    def counting(self, chunk, cursor, context, file):
        original(self, chunk, cursor, context, file)
        if context.get("read_basket", True) and context.get("basket_num") is not None:
            with lock:
                counts[(self._parent.name, context["basket_num"])] += 1

    model.read_members = counting
    try:
        yield counts
    finally:
        model.read_members = original


def common_offsets(branches):
    """Offsets shared by the branches that have baskets; ``None`` when none has any."""
    sets = [set(b.entry_offsets) for b in branches if b.num_baskets]
    return sorted(set.intersection(*sets)) if sets else None


def snap(boundary, offsets):
    return min(offsets, key=lambda o: (abs(o - boundary), o))


def even_bounds(n_entries, steps):
    return [(i * n_entries) // steps for i in range(steps + 1)]


def sized_bounds(n_entries, step_size):
    return [*range(0, n_entries, step_size), n_entries]


def aligned_ranges(bounds, offsets):
    """Non-empty ranges between the snapped ``bounds``."""
    snapped = [snap(b, offsets) for b in bounds]
    return [(a, b) for a, b in zip(snapped, snapped[1:]) if b > a]


def aligned_blind_lengths(n_entries, steps, offsets):
    """Per-step lengths, empty steps included, of ``steps`` snapped blind steps."""
    snapped = [snap(b, offsets) for b in even_bounds(n_entries, steps)]
    return [b - a for a, b in zip(snapped, snapped[1:])]
