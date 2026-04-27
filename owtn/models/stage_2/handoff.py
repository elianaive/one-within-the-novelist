"""Stage 1 → Stage 2 handoff models, plus Stage 2 → Stage 3 output schema.

Stage 1 emits champion files at `results/run_<ts>/stage_1/champions/island_*.json`
with `{id, code, metadata, combined_score}` shape (`code` is a JSON-string of the
ConceptGenome). This module loads those files into a typed `Stage1Winner` and
optionally enriches with per-match dimension votes from sibling `tournament.json`.

`Stage2Output` is the per-DAG record written into the handoff manifest at
`results/run_<ts>/stage_2/handoff_manifest.json`.

Standalone CLI:
    uv run python -m owtn.models.stage_2.handoff <path-to-champion.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG


class Stage1Winner(BaseModel):
    """One Stage 1 island champion as Stage 2 consumes it.

    Fields after `combined_score` are optional because they live in metadata
    (`affective_register`, `literary_mode`, `patch_type`) or in `tournament.json`
    (`tournament_rank`, `tournament_dimension_wins`) which is a sibling file —
    a champion JSON alone is a complete Stage 1 winner; tournament context is
    enrichment.
    """
    program_id: str
    genome: ConceptGenome
    combined_score: float
    affective_register: str | None = None
    literary_mode: str | None = None
    patch_type: str | None = None
    source_run: str | None = None
    tournament_rank: int | None = None
    tournament_dimension_wins: list[dict[str, Any]] | None = None

    @classmethod
    def from_champion_file(
        cls,
        path: Path,
        tournament_path: Path | None = None,
    ) -> Stage1Winner:
        """Load from `champions/island_*.json`, optionally enriching from tournament.json.

        If `tournament_path` is omitted, looks for `<champion_dir>/../tournament.json`.
        Missing tournament.json is non-fatal — `tournament_rank` / `tournament_dimension_wins`
        stay None.
        """
        champ = json.loads(path.read_text())
        program_id = champ["id"]
        genome = ConceptGenome.from_code_string(champ["code"])
        combined_score = float(champ["combined_score"])
        metadata = champ.get("metadata") or {}

        # Try sibling tournament.json by default — Stage 1 writes them in the
        # same `stage_1/` directory.
        if tournament_path is None:
            sibling = path.parent.parent / "tournament.json"
            tournament_path = sibling if sibling.exists() else None

        rank: int | None = None
        dim_wins: list[dict[str, Any]] | None = None
        if tournament_path is not None and tournament_path.exists():
            entries = json.loads(tournament_path.read_text())
            for entry in entries:
                if entry.get("program_id") == program_id:
                    rank = entry.get("rank")
                    dim_wins = entry.get("matches")
                    break

        # source_run = "run_<timestamp>" (parent of stage_1/)
        source_run: str | None = None
        try:
            source_run = path.parent.parent.parent.name
        except (AttributeError, IndexError):  # pragma: no cover
            pass

        return cls(
            program_id=program_id,
            genome=genome,
            combined_score=combined_score,
            affective_register=metadata.get("affective_register"),
            literary_mode=metadata.get("literary_mode"),
            patch_type=metadata.get("patch_type"),
            source_run=source_run,
            tournament_rank=rank,
            tournament_dimension_wins=dim_wins,
        )


class Stage2Output(BaseModel):
    """One advancing DAG in the Stage 2 → Stage 3 handoff manifest."""
    concept_id: str
    preset: str
    tournament_rank: int
    qd_cell: tuple[int, int]
    genome: DAG
    stage_1_forwarded: Stage1Winner
    mcts_reward: float
    adaptation_permissions: list[str] = Field(default_factory=list)


class Stage2HandoffManifest(BaseModel):
    """Per-run Stage 2 → Stage 3 handoff manifest.

    Written to `results/run_<ts>/stage_2/handoff_manifest.json` at run end.
    Stage 3 reads it to discover which structures advance and the metadata
    each carries.
    """
    run_id: str
    advancing: list[Stage2Output] = Field(default_factory=list)
    near_tie_promoted: bool = False  # set when near-tie boundary expansion fired


# ----- Standalone CLI -----

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Stage 1 champion JSON file and parse it as Stage1Winner. "
            "Tournament data (sibling tournament.json) loaded if present."
        ),
    )
    parser.add_argument("champion_path", type=Path, help="Path to champions/island_*.json")
    parser.add_argument(
        "--tournament", type=Path, default=None,
        help="Optional explicit path to tournament.json (defaults to sibling)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress success summary",
    )
    args = parser.parse_args(argv)

    if not args.champion_path.exists():
        print(f"error: {args.champion_path} not found", file=sys.stderr)
        return 1

    try:
        winner = Stage1Winner.from_champion_file(args.champion_path, args.tournament)
    except ValidationError as e:
        print(f"validation failed for {args.champion_path}:\n{e}", file=sys.stderr)
        return 1
    except (KeyError, json.JSONDecodeError) as e:
        print(f"malformed champion file {args.champion_path}: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        rank = winner.tournament_rank
        rank_str = str(rank) if rank is not None else "N/A (no tournament.json)"
        print(
            f"valid: program_id={winner.program_id[:12]}... "
            f"score={winner.combined_score:.3f} "
            f"rank={rank_str} "
            f"register={winner.affective_register} "
            f"mode={winner.literary_mode} "
            f"patch={winner.patch_type}"
        )
        print(
            f"  genome: anchor_role={winner.genome.anchor_scene.role!r} "
            f"premise[:60]={winner.genome.premise[:60]!r}..."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
