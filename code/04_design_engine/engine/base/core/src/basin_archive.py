"""Multimodal basin archive (blind-discovery repair).

Do not fit one diagonal Gaussian across disconnected elites.  This archive:

- clusters promising supported candidates in normalized (unit-cube) design
  space with a greedy radius rule;
- maintains several basin archives with a minimum allocation per basin;
- offers full-covariance local proposal distributions where numerically
  stable (Cholesky with diagonal fallback);
- enforces a novelty/global exploration quota every round;
- prevents one broad mediocre basin from consuming the entire population;
- logs basin IDs, archive membership, proposal sources, and survival reasons.
"""
from __future__ import annotations

import numpy as np


class BasinArchive:
    """Small, testable multimodal archive over unit-cube candidates."""

    def __init__(self, radius=0.12, min_per_basin=4, max_basins=8,
                 novelty_quota=0.15):
        if not (radius > 0):
            raise ValueError("radius must be positive")
        self.radius = float(radius)
        self.min_per_basin = int(min_per_basin)
        self.max_basins = int(max_basins)
        self.novelty_quota = float(novelty_quota)
        self.basins = []  # list of dicts: id, members, centre, cov, source_counts
        self.log = []

    def cluster(self, X, scores, supported, sources=None):
        """Greedy radius clustering, best-first, in normalized space."""
        X = np.asarray(X, dtype=float)
        scores = np.asarray(scores, dtype=float)
        supported = np.asarray(supported, dtype=bool)
        n = len(X)
        sources = list(sources) if sources is not None else ["unknown"] * n
        # Unsupported candidates are diagnostics only.  They must not create
        # proposal basins, otherwise a wholly unsupported cloud can consume
        # the local-search budget.
        order = sorted(np.flatnonzero(supported).tolist(),
                       key=lambda i: (-scores[i], i))
        basins = []
        for i in order:
            placed = None
            for b in basins:
                if float(np.linalg.norm(X[i] - b["centre"])) < self.radius:
                    placed = b
                    break
            if placed is None:
                basins.append({"id": len(basins), "members": [int(i)],
                               "centre": X[i].copy(),
                               "sources": {str(sources[i]): 1}})
            else:
                placed["members"].append(int(i))
                placed["sources"][str(sources[i])] = placed["sources"].get(str(sources[i]), 0) + 1
        # Recompute centres and full covariances.
        for b in basins:
            members = X[np.asarray(b["members"], dtype=int)]
            b["centre"] = members.mean(axis=0)
            if len(members) >= 3:
                try:
                    b["cov"] = np.cov(members.T) + 1e-6 * np.eye(X.shape[1])
                    b["cov_ok"] = True
                except Exception:
                    b["cov"] = np.diag(np.full(X.shape[1], 1e-4))
                    b["cov_ok"] = False
            else:
                b["cov"] = np.diag(np.full(X.shape[1], 1e-4))
                b["cov_ok"] = False
            b["n_supported"] = int(np.sum(supported[np.asarray(b["members"], dtype=int)]))
            b["best_score"] = float(np.max(scores[np.asarray(b["members"], dtype=int)]))
        # Keep the best basins, but never let one basin consume everything:
        # cap retained basins and guarantee minimum allocation later.
        basins.sort(key=lambda b: (-b["n_supported"], -b["best_score"], b["id"]))
        self.basins = basins[:self.max_basins]
        self.log.append({"event": "cluster", "n_basins": len(self.basins),
                         "basin_sizes": [len(b["members"]) for b in self.basins],
                         "basin_ids": [b["id"] for b in self.basins]})
        return self.basins

    def allocate(self, total):
        """Per-basin sample allocation with minimum guarantee + novelty quota."""
        total = int(total)
        if not self.basins:
            raise ValueError("allocate() requires cluster() first")
        n_novel = max(1, int(round(total * self.novelty_quota)))
        budget = max(total - n_novel, len(self.basins) * self.min_per_basin)
        per = [self.min_per_basin] * len(self.basins)
        rest = budget - sum(per)
        # Remaining budget goes to better basins, but no basin may exceed
        # half the non-novel budget (anti-monopoly cap).
        cap = max(self.min_per_basin, budget // 2)
        order = sorted(range(len(self.basins)),
                       key=lambda j: (-self.basins[j]["n_supported"],
                                      -self.basins[j]["best_score"], j))
        j = 0
        while rest > 0:
            b = order[j % len(order)]
            if per[b] < cap:
                per[b] += 1
                rest -= 1
            j += 1
            if j > 10 * len(order) + rest + 10:
                break
        alloc = {self.basins[j]["id"]: int(per[j]) for j in range(len(self.basins))}
        self.log.append({"event": "allocate", "total": total,
                         "novelty": int(n_novel), "per_basin": alloc,
                         "reason": "min-per-basin guarantee with anti-monopoly cap"})
        return alloc, int(n_novel)

    def propose(self, rng, per_basin_counts, dim):
        """Full-covariance local proposals per basin (whitened sampling)."""
        out, sources = [], []
        for b in self.basins:
            cnt = int(per_basin_counts.get(b["id"], 0))
            if cnt <= 0:
                continue
            try:
                chol = np.linalg.cholesky(b["cov"])
            except np.linalg.LinAlgError:
                chol = np.diag(np.sqrt(np.maximum(np.diag(b["cov"]), 1e-8)))
            z = rng.normal(size=(cnt, dim))
            block = b["centre"][None, :] + z @ chol.T
            out.append(np.clip(block, 0.0, 1.0))
            sources.extend([f"basin_{b['id']}"] * cnt)
        if not out:
            return np.empty((0, dim), dtype=float), []
        return np.concatenate(out, axis=0), sources

    def membership_rows(self):
        rows = []
        for b in self.basins:
            for m in b["members"]:
                rows.append({"basin_id": int(b["id"]), "member_index": int(m),
                             "basin_size": int(len(b["members"])),
                             "basin_sources": dict(b.get("sources", {}))})
        return rows
