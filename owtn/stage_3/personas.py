"""Voice-persona pool: typed Pydantic loader.

Reads the v3 persona YAMLs and validates schema. Each persona carries the
free-form aesthetic prose (identity, commitments, aversions, obsessions,
demonstrations) plus two structured fields the casting classifier needs:

- `exemplars` — corpus-passage references loaded into the persona's
  system prompt at session start (round-15 validated few-shots).
- `starved_by` — `[{tag, reason}]` patterns the casting classifier reads
  to determine whether the persona has anything to grip on for THIS concept.

The Pydantic model uses `extra="ignore"` so the parallel persona-iteration
agent can add fields (axis-position vectors, etc.) without breaking the
loader. Field reads are explicit; new fields are picked up by adding them
to the model.

Pool path: `lab/scratch/voice-persona-pool-v3/` until the parallel agent
migrates to `configs/voice_agents/`. Flip `DEFAULT_POOL_DIR` when that
lands.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL_DIR = REPO_ROOT / "lab" / "scratch" / "voice-persona-pool-v3"
"""V3 pool location. Flips to `configs/voice_agents/` when migration lands."""

DEFAULT_VOCABULARY_PATH = DEFAULT_POOL_DIR / "casting-vocabulary.yaml"
"""Closed vocabulary of concept-feature tags used by the casting classifier
and referenced by persona `starved_by` fields. Source of truth for tag
membership; persona tags outside this set are validation errors.

Lives in the pool dir for now (parallel agent's territory); flips to
`configs/voice_agents/casting-vocabulary.yaml` on migration."""


class Exemplar(BaseModel):
    """One reference-corpus passage attached to a persona as few-shot.

    `id` resolves against `data/voice-references/passages/`; loaded at
    session time (not at casting time — casting only needs the count and
    notes for trace context).
    """
    id: str = Field(min_length=2)
    note: str = Field(min_length=10)


class StarvationPattern(BaseModel):
    """One condition under which a persona has nothing to grip on.

    `tag` is a snake_case label following the convention
    `concept_<verb>_<property>` (verb ∈ requires|forbids|lacks|is|has).
    `reason` is a one-sentence explanation tied to the persona's
    load-bearing commitment — the classifier reads `reason` alongside the
    persona's first commitment to disambiguate firing direction.
    """
    tag: str = Field(min_length=4)
    reason: str = Field(min_length=10)


class VoicePersona(BaseModel):
    """One voice-agent persona — pool member.

    Casting reads `id`, `name`, `aesthetic_commitments[0]` (load-bearing
    first commitment), `identity` (for engagement disambiguation), and
    `starved_by` (per-pattern firing checks). Other fields are carried
    through for downstream phase implementations.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = Field(min_length=2)
    name: str = Field(min_length=2)
    identity: str = Field(min_length=40)
    aesthetic_commitments: list[str] = Field(min_length=1)
    aversions: list[str] = Field(default_factory=list)
    obsessions: list[str] = Field(default_factory=list)
    epistemic_skepticism: str = Field(min_length=20)
    demonstrations: list[str] = Field(default_factory=list)
    exemplars: list[Exemplar] = Field(default_factory=list)
    starved_by: list[StarvationPattern] = Field(default_factory=list)
    model: list[str] = Field(default_factory=lambda: ["deepseek-v4-pro"])
    temperature: float = 0.8
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None

    @property
    def first_commitment(self) -> str:
        """The load-bearing aesthetic commitment — used by the casting
        classifier to evaluate engagement."""
        return self.aesthetic_commitments[0]


def load_persona_pool(
    pool_dir: Path | str | None = None,
) -> list[VoicePersona]:
    """Load persona YAMLs from the pool directory.

    The pool directory is expected to contain persona YAMLs only — once
    migrated to `configs/voice_agents/`, vocabulary and other registries
    live elsewhere. Malformed YAMLs are logged and skipped without
    killing the pool.

    Returns personas sorted by id for deterministic ordering.
    """
    pool_dir = Path(pool_dir) if pool_dir else DEFAULT_POOL_DIR
    if not pool_dir.is_dir():
        logger.warning("persona pool dir does not exist: %s", pool_dir)
        return []

    personas: list[VoicePersona] = []
    for path in sorted(pool_dir.glob("*.yaml")):
        # Skip co-located registries that share the YAML extension but
        # aren't personas (vocabulary, _template, etc.). Anything that
        # doesn't have a top-level `id` field is treated as non-persona
        # config and silently skipped — keeps the pool dir multi-purpose
        # without spamming validation errors at startup.
        if path.stem.startswith("_") or path.name == "casting-vocabulary.yaml":
            continue
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            logger.warning("persona YAML unreadable: %s (%s)", path, e)
            continue
        if not isinstance(raw, dict):
            logger.warning("persona YAML not a mapping: %s", path)
            continue
        if "id" not in raw:
            continue  # non-persona registry file
        try:
            persona = VoicePersona.model_validate(raw)
        except Exception as e:
            logger.warning("persona %s failed validation: %s", path.name, e)
            continue
        personas.append(persona)

    personas.sort(key=lambda p: p.id)
    if not personas:
        logger.warning("persona pool is empty after load: %s", pool_dir)
    return personas


def load_casting_vocabulary(
    path: Path | str | None = None,
) -> dict[str, str]:
    """Load the closed vocabulary of concept-feature tags.

    Returns `{tag: meaning}`. The casting classifier produces a subset
    of these tags as TRUE for each concept; persona `starved_by` tags
    must be members of this vocabulary. Returns an empty dict if the
    file is missing or unparseable (caller decides the policy).
    """
    p = Path(path) if path else DEFAULT_VOCABULARY_PATH
    if not p.exists():
        logger.warning("casting vocabulary not found at %s", p)
        return {}
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        logger.warning("casting vocabulary unreadable: %s (%s)", p, e)
        return {}
    if not isinstance(raw, dict):
        logger.warning("casting vocabulary not a mapping: %s", p)
        return {}
    entries = raw.get("vocabulary", [])
    out: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tag = entry.get("tag")
        meaning = entry.get("meaning", "")
        if tag and isinstance(tag, str):
            out[tag] = meaning
    if not out:
        logger.warning("casting vocabulary is empty: %s", p)
    return out


def validate_pool_against_vocabulary(
    pool: list[VoicePersona],
    vocabulary: dict[str, str],
) -> list[str]:
    """Return a list of validation errors — empty list means pool is clean.

    Each persona's `starved_by` tags must be members of the vocabulary.
    Caller decides whether to fail-fast or warn-and-continue on errors.
    """
    errors: list[str] = []
    for p in pool:
        for sb in p.starved_by:
            if sb.tag not in vocabulary:
                errors.append(
                    f"{p.id}: starved_by tag {sb.tag!r} not in vocabulary"
                )
    return errors
