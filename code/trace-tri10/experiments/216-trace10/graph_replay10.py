"""Exp 216 (trust base): exact graph reconstruction for n = 10 with clique-range slices.
Copy of exp 160 graph_replay (independent, cached forbidden-third-cell masks) plus:
  graph_cells(support, q, grid, threshold) -> the kept cell domains, in the checker's own order;
  graph_boxes(..., first_range=(a, b)) -> only the cliques whose first (smallest) cell index lies in [a, b), in the
  same order as the full enumeration. Slices [0, c1), [c1, c2), ..., [ck, count) therefore enumerate every clique
  exactly once."""
import hashlib, os, time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from itertools import combinations

import numpy as np

from verify_trace import det_range, keys, p2_bound, require, triple_at, vertices


def vertex_array(box, q):
    corners = vertices(box, q)
    require(bool(corners), "empty graph point domain")
    return np.asarray(corners + [corners[0]] * (5 - len(corners)), dtype=np.int64)


def blocks(a, b, candidates, r):
    u = b[None, :, :] - a[:, None, :]
    v = candidates[:, None, :, :] - a[None, :, None, :]
    d = u[None, :, :, None, 0] * v[:, :, None, :, 1] - u[None, :, :, None, 1] * v[:, :, None, :, 0]
    return (d.min(axis=(1, 2, 3)) >= -r) & (d.max(axis=(1, 2, 3)) <= r)


def mask(flags):
    return int.from_bytes(np.packbits(flags, bitorder="little").tobytes(), "little")


def _prepare(support, q, grid, threshold):
    require(type(grid) is int and 1 <= grid <= q and q % grid == 0
            and Fraction(2, grid) <= 2 * threshold, "cell occupancy condition failed")
    require(2 * q * q < 2 ** 63, "integer determinant overflow bound failed")
    require(0 < 2 * threshold <= 2, "threshold outside accelerated range")
    r = int(2 * threshold * q * q)
    width = q // grid
    domains = [("R", x * width, (x + 1) * width, y * width, (y + 1) * width)
               for x in range(grid) for y in range(grid - x)]
    xyz = np.asarray([vertex_array(d, q) for d in domains])
    supports = [vertex_array(d, q) for d in support]
    keep = np.ones(len(domains), dtype=bool)
    for a, b in combinations(supports, 2):
        keep &= ~blocks(a, b, xyz, r)
    cells = [cell for cell, kept in zip(domains, keep.tolist()) if kept]
    return r, cells, xyz[keep], supports


def graph_cells(support, q, grid, threshold):
    return _prepare(support, q, grid, threshold)[1]


class _Graph:
    """The cell graph of one support box: cells, edges and memoised forbidden-third-cell masks. A pure function
    of (support, q, grid, threshold); cached so that the shards of one support (e.g. clique slices) share it."""
    def __init__(self, support, q, grid, threshold):
        self.r, self.cells, self.xyz, self.supports = _prepare(support, q, grid, threshold)
        count = len(self.cells)
        self.edges = []
        for i in range(count):
            keep = np.arange(count) > i
            for a in self.supports:
                keep &= ~blocks(a, self.xyz[i], self.xyz, self.r)
            self.edges.append(mask(keep))
        self._forbidden = {}
        self._excl = {}
        self._pair_edges = {}             # exp 223: edge lists with inherited pairs removed
        # Optional shared mask cache (env REPLAY_MASK_CACHE, set by verify_manifest10 for parallel replays): masks
        # this checker computed in sibling worker processes on this machine, as text lines "a b hexmask" in one
        # folder per (support, q, grid, threshold). Only complete files are read (written as .tmp, then renamed).
        # A clean-room replay simply leaves the variable unset and recomputes every mask.
        self._pending, self._seen, self._dir = {}, set(), None
        root = os.environ.get("REPLAY_MASK_CACHE")
        if root:
            key = hashlib.sha256(repr((tuple(support), q, grid, str(threshold))).encode()).hexdigest()[:24]
            self._dir = Path(root) / key; self._dir.mkdir(parents=True, exist_ok=True); self.sync()

    def sync(self):
        """publish masks computed here, then load the ones sibling workers published"""
        if self._dir is None: return
        if self._pending:
            name = f"{os.getpid()}-{time.time_ns()}.jsonl"; tmp = self._dir / (name + ".tmp")
            tmp.write_text("".join(f"{a} {b} {m:x}\n" for (a, b), m in self._pending.items()), encoding="ascii")
            tmp.replace(self._dir / name); self._seen.add(name); self._pending = {}
        for f in sorted(self._dir.glob("*.jsonl")):
            if f.name in self._seen: continue
            self._seen.add(f.name)
            for line in f.read_text(encoding="ascii").splitlines():
                a, b, h = line.split(); self._forbidden.setdefault((int(a), int(b)), int(h, 16))

    def forbidden(self, a, b):
        key = (a, b)
        m = self._forbidden.get(key)
        if m is None:
            m = self._forbidden[key] = mask(blocks(self.xyz[a], self.xyz[b], self.xyz, self.r))
            if self._dir is not None:
                self._pending[key] = m
                if len(self._pending) >= 2000: self.sync()
        return m


@lru_cache(maxsize=4)
def _graph(support, q, grid, threshold):
    return _Graph(support, q, grid, threshold)


def exclusions(G, support, q, threshold, conflicts):
    """Verified single-cell exclusions. A record {"cells": [i], "weights": [[t, w], ...]} or {"cells": [i],
    "triple": t} proves, by the P2 lemma or a direct bound on the k + 1 point box support + (cell i,), that no
    configuration with every area > t has a free point in cell i (the k + 1 points' triangles are a subset of any
    full configuration's). Returns the bit mask of excluded cells."""
    require(type(conflicts) is list, "invalid graph witnesses")
    r = int(2 * threshold * q * q); ex = 0; prior = -1
    for rec in conflicts:
        require(type(rec) is dict and "cells" in rec, "invalid exclusion record")
        ids = rec["cells"]
        require(type(ids) is list and len(ids) == 1 and type(ids[0]) is int and prior < ids[0] < len(G.cells),
                "invalid exclusion cell")
        i = prior = ids[0]
        box = tuple(support) + (G.cells[i],)
        if "weights" in rec:
            keys(rec, "cells weights")
            require(p2_bound(box, rec["weights"], q, threshold) <= threshold, "exclusion P2 inequality failed")
        else:
            keys(rec, "cells triple")
            lo, hi = det_range(box, triple_at(rec["triple"], box), q)
            require(-r <= lo <= hi <= r, "exclusion bound failed")
        ex |= 1 << i
    return ex


def grid_cell(d, q, grid):
    """d must be one closed cell of the M-grid inside the triangle: ("R", x w, (x+1) w, y w, (y+1) w), x + y < M"""
    require(type(d) is list and len(d) == 5 and d[0] == "R" and all(type(v) is int for v in d[1:]), "invalid pair cell")
    w = q // grid; x, y = d[1] // w, d[3] // w
    require(list(d) == ["R", x * w, (x + 1) * w, y * w, (y + 1) * w] and 0 <= x and 0 <= y and x + y <= grid - 1,
            "pair cell is not a grid cell")
    return tuple(d)


def pair_exclusions(support, q, grid, threshold, records):
    """Exp 223: verified cell-pair exclusions at a support box. A record {"cells": [c1, c2], "weights": [[t, w],
    ...]} or {"cells": [c1, c2], "triple": t} (c1 < c2 grid cells, records strictly increasing) proves, by the P2
    lemma or a direct bound on the k + 2 point box support + (c1, c2), that no configuration with every area > t
    and support in this box has free points in both cells (those k + 2 points' triangles are a subset of the full
    configuration's). Every configuration of a sub-box of `support` is a configuration of `support`, so the
    returned set of pairs stays valid in all descendants of the partition node that carries it."""
    require(type(records) is list, "invalid pair exclusions")
    require(type(grid) is int and 1 <= grid <= q and q % grid == 0, "invalid pair grid")   # audit hardening
    support = tuple(tuple(d) for d in support); threshold = Fraction(threshold)
    r = int(2 * threshold * q * q); out, prior = set(), None
    for rec in records:
        require(type(rec) is dict and "cells" in rec and type(rec["cells"]) is list and len(rec["cells"]) == 2,
                "invalid pair exclusion record")
        c1, c2 = (grid_cell(c, q, grid) for c in rec["cells"])
        require(c1 < c2 and (prior is None or prior < (c1, c2)), "pair exclusions must be strictly increasing")
        prior = (c1, c2)
        box = support + (c1, c2)
        if "weights" in rec:
            keys(rec, "cells weights")
            require(p2_bound(box, rec["weights"], q, threshold) <= threshold, "pair exclusion P2 inequality failed")
        else:
            keys(rec, "cells triple")
            lo, hi = det_range(box, triple_at(rec["triple"], box), q)
            require(-r <= lo <= hi <= r, "pair exclusion bound failed")
        out.add((c1, c2))
    return frozenset(out)


def graph_boxes(support, q, grid, n, threshold, conflicts, first_range=None, pairs=frozenset(), pairs_key=""):
    """pairs: cell pairs (c1 < c2) already verified by pair_exclusions at an ancestor support box; their edges are
    removed. pairs_key identifies the set for caching (the digests of its files)."""
    G = _graph(tuple(tuple(d) for d in support), q, grid, Fraction(threshold))
    G.sync()                                             # share masks between shards / workers (no-op without cache)
    key = repr(conflicts)
    ex = G._excl.get(key)
    if ex is None:
        ex = G._excl[key] = exclusions(G, tuple(tuple(d) for d in support), q, Fraction(threshold), conflicts)
    cells, edges, forbidden = G.cells, G.edges, G.forbidden
    if pairs:
        pk = pairs_key or repr(sorted(pairs))
        cut = G._pair_edges.get(pk)
        if cut is None:
            index = {c: i for i, c in enumerate(cells)}
            cut = list(edges)
            for c1, c2 in pairs:
                i, j = index.get(c1), index.get(c2)
                if i is not None and j is not None:
                    i, j = min(i, j), max(i, j); cut[i] &= ~(1 << j)
            G._pair_edges[pk] = cut
        edges = cut
    count = len(cells)
    a0, b0 = (0, count) if first_range is None else first_range
    require(type(a0) is int and type(b0) is int and 0 <= a0 < b0 <= count, "invalid clique slice")

    def visit(selected, candidates):
        need = n - len(support) - len(selected)
        if need == 0:
            yield support + tuple(cells[i] for i in selected)
            return
        while candidates.bit_count() >= need:
            bit = candidates & -candidates
            candidates ^= bit
            i = bit.bit_length() - 1
            children = candidates & edges[i]
            if need > 1:
                for j in selected:
                    if children.bit_count() < need - 1:
                        break
                    children &= ~forbidden(j, i)
            yield from visit(selected + (i,), children)

    need0 = n - len(support)
    require(need0 >= 1, "no free points")

    def sliced():
        # top level of visit((), all cells) restricted to first index in [a0, b0): for first i the remaining
        # candidates are exactly the cells above i, so children = edges[i]; identical order to the full walk.
        for i in range(a0, b0):
            if (ex >> i) & 1:
                continue                     # excluded cell: no clique starts (or continues) here
            if need0 == 1:
                yield support + (cells[i],)
            elif (count - 1 - i) >= need0 - 1:
                yield from visit((i,), edges[i] & ~ex)

    return sliced()
