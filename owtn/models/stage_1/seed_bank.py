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


class SeedBank:
    def __init__(self, seeds: list[Seed]) -> None:
        self.seeds = seeds
        self._by_type: dict[str, list[Seed]] = {}
        for s in seeds:
            self._by_type.setdefault(s.type, []).append(s)

    @classmethod
    def load(cls, path: str) -> SeedBank:
        with open(Path(path)) as f:
            raw = yaml.safe_load(f)
        seeds = [Seed.model_validate(entry) for entry in raw]
        return cls(seeds)

    def get_by_type(self, seed_type: str) -> list[Seed]:
        return self._by_type.get(seed_type, [])

    def select(
        self,
        seed_types: str | list[str],
        exclude_ids: set[str] | None = None,
    ) -> Seed | None:
        """Pick a random seed pooled uniformly across the given type(s)."""
        types = [seed_types] if isinstance(seed_types, str) else seed_types
        candidates = [s for t in types for s in self.get_by_type(t)]
        if exclude_ids:
            candidates = [s for s in candidates if s.id not in exclude_ids]
        if not candidates:
            return None
        return random.choice(candidates)
