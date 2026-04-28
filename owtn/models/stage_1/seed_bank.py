from __future__ import annotations

import random
from pathlib import Path

import yaml
from pydantic import BaseModel


# implementation.md lines 313-326
SEED_OPERATOR_MAP: dict[str, list[str]] = {
    "real_world": ["real_world_seed"],
    "thought_experiment": ["thought_experiment"],
    "axiom": ["thought_experiment"],
    "dilemma": ["thought_experiment", "collision", "compression"],
    "constraint": ["constraint_first"],
    "noun_cluster": ["noun_list"],
    "image": ["discovery"],
    "compression": ["compression"],
    "collision_pair": ["collision"],
    "anti_target": ["anti_premise"],
}

# Reverse: operator -> seed types
OPERATOR_SEED_TYPES: dict[str, list[str]] = {}
for seed_type, operators in SEED_OPERATOR_MAP.items():
    for op in operators:
        OPERATOR_SEED_TYPES.setdefault(op, []).append(seed_type)


class Seed(BaseModel):
    id: str
    type: str
    content: str | list[str]
    source: str | None = None
    tags: list[str]


# Top-K used by `select_farthest_first` to preserve some stochasticity in
# what's otherwise a deterministic max-min pick. K=3 is a conservative
# middle: still farthest-first in expectation, but two runs starting from
# the same cold state aren't identical. Justified in the issue.
_FARTHEST_FIRST_TOP_K = 3


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance on already-normalized vectors falls out to
    1 - dot product. Embeddings from `seed_embeddings.compute_embeddings`
    are L2-normalized, so we skip the norm in the hot path."""
    return 1.0 - sum(x * y for x, y in zip(a, b))


class SeedBank:
    def __init__(self, seeds: list[Seed]) -> None:
        self.seeds = seeds
        self._by_type: dict[str, list[Seed]] = {}
        for s in seeds:
            self._by_type.setdefault(s.type, []).append(s)
        self._embeddings: dict[str, list[float]] | None = None

    @classmethod
    def load(cls, path: str) -> SeedBank:
        with open(Path(path)) as f:
            raw = yaml.safe_load(f)
        seeds = [Seed.model_validate(entry) for entry in raw]
        return cls(seeds)

    def attach_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        """Provide unit-norm embeddings keyed by seed id. Needed for
        `select_farthest_first`; ignored by `select`."""
        self._embeddings = embeddings

    def get_by_type(self, seed_type: str) -> list[Seed]:
        return self._by_type.get(seed_type, [])

    def _candidates(
        self, seed_types: str | list[str], exclude_ids: set[str] | None
    ) -> list[Seed]:
        types = [seed_types] if isinstance(seed_types, str) else seed_types
        candidates = [s for t in types for s in self.get_by_type(t)]
        if exclude_ids:
            candidates = [s for s in candidates if s.id not in exclude_ids]
        return candidates

    def select(
        self,
        seed_types: str | list[str],
        exclude_ids: set[str] | None = None,
    ) -> Seed | None:
        """Uniform random pick across the union of the given type pools."""
        candidates = self._candidates(seed_types, exclude_ids)
        return random.choice(candidates) if candidates else None

    def select_farthest_first(
        self,
        seed_types: str | list[str],
        used_ids: set[str],
    ) -> Seed | None:
        """Pick a seed maximally unlike already-used ones (k-center heuristic).

        Distance is cosine on attached embeddings. The candidate's score is
        `min(distance to any used seed)`; we sort by descending score and
        sample uniformly from the top-K to preserve some stochasticity. With
        no used seeds (cold start) or no embeddings attached, falls back to
        uniform random.

        Already-used seeds are filtered out — the caller's `used_ids` is the
        excluded set.
        """
        candidates = self._candidates(seed_types, used_ids)
        if not candidates:
            return None
        if not used_ids or self._embeddings is None:
            return random.choice(candidates)
        used_embs = [self._embeddings[uid] for uid in used_ids if uid in self._embeddings]
        if not used_embs:
            return random.choice(candidates)
        scored: list[tuple[float, Seed]] = []
        for seed in candidates:
            emb = self._embeddings.get(seed.id)
            if emb is None:
                continue
            min_dist = min(_cosine_distance(emb, ue) for ue in used_embs)
            scored.append((min_dist, seed))
        if not scored:
            return random.choice(candidates)
        scored.sort(key=lambda x: -x[0])
        top = scored[: min(_FARTHEST_FIRST_TOP_K, len(scored))]
        return random.choice([s for _, s in top])
