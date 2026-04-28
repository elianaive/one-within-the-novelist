"""Stage 1 concept evolution runner — thin wrapper around ShinkaEvolveRunner.

Configures the loop for JSON concept genomes, dispatches through the operator
registry, and overrides initial population generation to sample from the
genesis-eligible subset of operators.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.config import EvolutionConfig as ShinkaEvolutionConfig, FOLDER_PREFIX
from shinka.core.sampler import PromptSampler
from shinka.database import DatabaseConfig as ShinkaDatabaseConfig, Program
from shinka.edit.async_apply import write_file_async
from shinka.launch import LocalJobConfig
from shinka.llm import extract_between

from owtn.evaluation.pairwise import compare as pairwise_compare
from owtn.evaluation.tournament import run_tournament
from owtn.llm.call_logger import llm_context, llm_log_dir
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_1.config import StageConfig
from owtn.models.stage_1.seed_bank import SeedBank
from owtn.prompts import sample_tonal_steering
from owtn.prompts.stage_1.registry import (
    build_operator_prompt,
    filter_genesis_eligible,
    load_registry,
)
from owtn.state_logger import snapshot_generation

logger = logging.getLogger(__name__)

def _build_shinka_configs(
    cfg: StageConfig,
    config_path: str,
) -> tuple[ShinkaEvolutionConfig, ShinkaDatabaseConfig, LocalJobConfig]:
    """Translate our StageConfig into ShinkaEvolve dataclass configs."""
    # Per-model generation params: flatten list[GenerationModelConfig] into
    # parallel lists indexed by model. sample_model_kwargs picks a model by
    # weight, then uses the same index across all parallel lists to build a
    # consistent per-model kwargs bundle.
    gm = cfg.llm.generation_models
    llm_kwargs = {
        "temperatures": [m.temperature for m in gm],
        "max_tokens": 16384,
        "reasoning_efforts": [m.reasoning_effort for m in gm],
        "thinking_tokens": [m.thinking_tokens for m in gm],
        "top_p": [m.top_p for m in gm],
        "top_k": [m.top_k for m in gm],
        "min_p": [m.min_p for m in gm],
    }
    evo = ShinkaEvolutionConfig(
        task_sys_msg=None,  # set by PromptSampler via registry
        language=cfg.evolution.language,
        patch_types=cfg.evolution.patch_types,
        patch_type_probs=cfg.evolution.patch_type_probs,
        num_generations=cfg.evolution.num_generations,
        max_patch_resamples=cfg.evolution.max_patch_resamples,
        llm_models=[m.name for m in gm],
        llm_kwargs=llm_kwargs,
        llm_dynamic_selection=cfg.evolution.llm_dynamic_selection,
        use_text_feedback=cfg.evolution.use_text_feedback,
        evolve_prompts=cfg.evolution.evolve_prompts,
        meta_rec_interval=cfg.evolution.meta_rec_interval,
        code_embed_sim_threshold=cfg.evolution.code_embed_sim_threshold,
        max_novelty_attempts=cfg.evolution.max_novelty_attempts,
        embedding_model=cfg.llm.embedding_model,
    )

    db = ShinkaDatabaseConfig(
        num_islands=cfg.database.num_islands,
        archive_size=cfg.database.archive_size,
        archive_selection_strategy=cfg.database.archive_selection_strategy,
        migration_interval=cfg.database.migration_interval,
        migration_rate=cfg.database.migration_rate,
        island_elitism=cfg.database.island_elitism,
        island_selection_strategy=cfg.database.island_selection_strategy,
        enable_dynamic_islands=cfg.database.enable_dynamic_islands,
        stagnation_threshold=cfg.database.stagnation_threshold,
        parent_selection_strategy=cfg.database.parent_selection_strategy,
    )

    # eval_function is set in __init__ after DB is available (needs self.db).
    job = LocalJobConfig(
        extra_cmd_args={"config_path": config_path},
    )

    return evo, db, job


# Epsilon for champion succession. Constraints:
#   - small enough that hundreds of cumulative successions stay well below
#     the score precision floor relevant to archive/threshold logic
#     (10^4 generations × 1e-6 = 0.01, still <<1% of a typical 0.5-1.0 score
#     and far below any threshold currently configured)
#   - large enough that float64 comparison and SQL ORDER BY treat the new
#     champion as strictly greater than the deposed incumbent (well above
#     ~2.2e-16 machine epsilon for values near 1.0)
_SUCCESSION_EPS = 1e-6


def _challenger_succession_score(
    *, match_score: float, incumbent_score: float
) -> float:
    """Score assigned to a challenger that has won a pairwise match.

    Guarantees the new champion outranks the deposed incumbent in
    `get_island_champion`'s ORDER BY combined_score DESC sort. Without
    this, an early decisive upset (e.g. 8-1-0 = 0.89 against a weak seed)
    creates a permanent moat that later challengers can beat in pairwise
    but never dethrone in selection. See
    `lab/issues/2026-04-18-champion-succession-score-bug.md`.
    """
    return max(match_score, incumbent_score + _SUCCESSION_EPS)


class ConceptEvolutionRunner(ShinkaEvolveRunner):
    """ShinkaEvolve runner configured for Stage 1 concept evolution."""

    def __init__(
        self,
        config_path: str = "configs/stage_1/medium.yaml",
        *,
        verbose: bool = True,
        max_evaluation_jobs: int = 2,
        max_proposal_jobs: int = 1,
        results_dir: Optional[str] = None,
    ):
        self.stage_config = StageConfig.from_yaml(config_path)
        evo, db, job = _build_shinka_configs(self.stage_config, config_path)
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = str(Path("results") / f"run_{timestamp}" / "stage_1")
        evo.results_dir = results_dir

        self._log_dir = results_dir

        # Load seed bank and operator registry
        self.seed_bank = SeedBank.load(self.stage_config.paths.seed_bank)
        self.registry = load_registry()

        # Per-run set of seed ids that have been injected. Mutated by
        # `inject_seed` via `build_operator_prompt(used_seed_ids=...)`. Under
        # `seed_sampling_strategy="farthest_first"` this is also the reference
        # set for distance scoring. Under "uniform" it just acts as the
        # exclude list (don't reuse a seed within a run).
        self.used_seed_ids: set[str] = set()
        if self.stage_config.seed_sampling_strategy == "farthest_first":
            from owtn.models.stage_1.seed_embeddings import load_or_compute
            cache_path = Path(self.stage_config.paths.seed_embeddings_cache)
            embeddings = load_or_compute(self.seed_bank.seeds, cache_path)
            self.seed_bank.attach_embeddings(embeddings)

        # Per-island lock guarding the post-pairwise champion-file
        # read+update phase. Pairwise comparisons run in parallel (the slow
        # LLM calls overlap), but the brief "did I dethrone?" handoff must
        # serialize: without it, two parallel winners both read the
        # pre-update incumbent score, both compute `incumbent + ε`, and
        # both end up with the same score — breaking the succession chain.
        # See lab/issues/2026-04-19-parallel-eval-champion-race.md.
        #
        # Pre-allocate locks for the static island count; dynamic islands
        # (enable_dynamic_islands) lazy-create under a creation meta-lock
        # via `_get_champion_lock`. Must not use defaultdict(Lock) — its
        # __missing__ insertion is not safe under concurrent access.
        self._champion_locks: dict[int, threading.Lock] = {
            i: threading.Lock()
            for i in range(self.stage_config.database.num_islands)
        }
        self._champion_locks_creation = threading.Lock()

        # Run-wide population brief. Recomputed end-of-gen from all cached
        # lineage briefs; injected into next gen's mutation prompts as
        # ambient pressure. Split into two blocks: context (middle of prompt)
        # and exploration_directions (top + end — instruction sandwich for
        # primacy + recency). None until the first end-of-gen tick fires, or
        # indefinitely if LLMConfig.run_brief_model is not set.
        self._latest_population_context: str | None = None
        self._latest_exploration_directions: str | None = None

        super().__init__(
            evo_config=evo,
            job_config=job,
            db_config=db,
            verbose=verbose,
            max_evaluation_jobs=max_evaluation_jobs,
            max_proposal_jobs=max_proposal_jobs,
        )

        # Replace the prompt_sampler with one that knows about operators + seeds
        self.prompt_sampler = PromptSampler(
            task_sys_msg=None,
            language="json",
            patch_types=evo.patch_types,
            patch_type_probs=evo.patch_type_probs,
            use_text_feedback=evo.use_text_feedback,
            seed_bank=self.seed_bank,
            genesis_ratio=self.stage_config.evolution.genesis_ratio,
            prompt=self.stage_config.prompt,
        )
        self.prompt_sampler.tonal_inherit_rate = self.stage_config.evolution.tonal_inherit_rate
        self.prompt_sampler.tonal_crossover_new_rate = self.stage_config.evolution.tonal_crossover_new_rate
        # Mutation path uses the same shared used-seed-set as the cold-start
        # path so seeds aren't re-picked across genesis and mutation calls.
        self.prompt_sampler.seed_sampling_strategy = self.stage_config.seed_sampling_strategy
        self.prompt_sampler.used_seed_ids = self.used_seed_ids

        # Set eval_function now that self.db is available (from super().__init__).
        # The eval function reads champion from a file (not DB) to avoid
        # SQLite threading issues — the scheduler runs it in a thread pool.
        self._champions_dir = Path(self.results_dir) / "champions"
        self._champions_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler.config.eval_function = self._evaluate_with_pairwise

        # Opt-in critique-revise cycle for configured generators. Fires after
        # every generation/genesis/mutation call on a listed model. Value is
        # the critic-call reasoning_effort override; "disabled" (the default)
        # strips thinking kwargs so the critic isn't re-thinking the same
        # ground the generator already covered.
        from owtn.optimizer.self_critic import register_self_critic_models
        register_self_critic_models({
            m.name: m.self_critic_reasoning_effort
            for m in self.stage_config.llm.generation_models
            if m.self_critic
        })

    def _write_champions_to_disk(self) -> None:
        """Write each island's current champion genome to disk.

        Called from the main thread before evaluation jobs run, so the
        eval function (in a worker thread) can read champions without
        touching the database.
        """
        num_islands = self.stage_config.database.num_islands
        for island_idx in range(num_islands):
            champion = self.db.get_island_champion(island_idx)
            if champion is None:
                # Remove stale champion file.
                champ_file = self._champions_dir / f"island_{island_idx}.json"
                champ_file.unlink(missing_ok=True)
                continue
            champ_file = self._champions_dir / f"island_{island_idx}.json"
            champ_file.write_text(json.dumps({
                "id": champion.id,
                "code": champion.code,
                "metadata": champion.metadata or {},
                "combined_score": champion.combined_score,
            }))

    def _get_champion_lock(self, island_idx: int) -> threading.Lock:
        """Return the per-island champion-update lock, creating it if the
        island appeared at runtime via enable_dynamic_islands.

        Double-checked under `_champion_locks_creation` so concurrent calls
        for the same dynamic island can't construct competing Lock objects.
        """
        lock = self._champion_locks.get(island_idx)
        if lock is not None:
            return lock
        with self._champion_locks_creation:
            return self._champion_locks.setdefault(island_idx, threading.Lock())

    def _evaluate_with_pairwise(
        self, program_path: str, results_dir: str, config_path: str,
        parent_id: str | None = None, island_idx: int | None = None,
    ) -> None:
        """Inline evaluation: validate + pairwise in one blocking step.

        Called by ShinkaEvolve's scheduler in a worker thread. Validates
        the genome, reads the champion from disk (written by main thread),
        runs pairwise comparison, and writes metrics.json. Blocks until
        pairwise completes, so ShinkaEvolve waits for the real score.

        Per-island serialization: only one eval runs against an island's
        champion at a time. Parallelism is preserved *across* islands —
        with 2 islands and `--max-eval-jobs=2`, two challengers can
        evaluate simultaneously as long as they're on different islands.
        Without this, two parallel challengers on the same island would
        both read the same incumbent score, both succeed via succession
        with the identical `incumbent + ε` value, and both append a
        defense critique to the same champion. See
        `lab/issues/2026-04-19-parallel-eval-champion-race.md`.

        ``island_idx`` is required: ShinkaEvolve's scheduler always passes
        it when calling this eval function. A None island_idx would mean
        the per-island lock can't be acquired and the race-fix above is
        defeated, so we fail loudly rather than silently skipping the lock.
        """
        if island_idx is None:
            raise RuntimeError(
                "_evaluate_with_pairwise called without island_idx — would "
                "skip the per-island champion lock. Check scheduler call site."
            )

        import asyncio

        # Propagate context vars to this worker thread.
        llm_log_dir.set(self._log_dir)
        llm_context.set({"role": "pairwise_judge"})

        # Single asyncio.run() for all async work in this eval — avoids
        # httpx/aiohttp cleanup errors from loop creation/destruction.
        with self._get_champion_lock(island_idx):
            asyncio.run(self._evaluate_with_pairwise_async(
                program_path, results_dir, config_path, island_idx,
            ))

    def _read_champion_data(self, island_idx: int) -> dict | None:
        """Read the island's champion JSON from disk. Returns None if no
        champion file exists or the file can't be parsed (the caller treats
        either case as "this is the island's first concept")."""
        champ_file = self._champions_dir / f"island_{island_idx}.json"
        if not champ_file.exists():
            return None
        try:
            return json.loads(champ_file.read_text())
        except Exception:
            return None

    def _apply_pairwise_outcome(
        self,
        *,
        result,
        pairwise_result,
        champion_data: dict,
        program_path: str,
        island_idx: int,
    ) -> None:
        """Score the challenger based on the pairwise verdict, mutating
        `result` in place. Winner branch also writes the new champion file
        immediately (the per-island lock guarantees that's race-free)."""
        result.text_feedback = pairwise_result.feedback
        if pairwise_result.winner == "a":
            # Eval is serialized per island via _champion_locks[island_idx]
            # (see _evaluate_with_pairwise), so the champion_data we read at
            # the start of this eval still reflects the actual incumbent —
            # no parallel challenger snuck in.
            incumbent_score = float(champion_data.get("combined_score", 0.5))
            result.combined_score = _challenger_succession_score(
                match_score=pairwise_result.a_score,
                incumbent_score=incumbent_score,
            )
            # Update champion file immediately so the next eval on this
            # island (which will block on the same lock) sees the new
            # champion when it eventually runs.
            champ_file = self._champions_dir / f"island_{island_idx}.json"
            champ_file.write_text(json.dumps({
                "id": "pending",
                "code": Path(program_path).read_text(),
                "metadata": {},
                "combined_score": result.combined_score,
            }))
            verdict = "WINS"
        else:
            # combined_score drives shinka's parent-selection for losers.
            # Use a_weighted_score (not a_score) so parent-selection matches
            # the winner-selection basis — both driven by weighted dim-votes.
            result.combined_score = pairwise_result.a_weighted_score * 0.9
            verdict = "LOSES"

        logger.info(
            "%s (%d-%d-%d, weighted %.2f-%.2f, score=%.0f%%)",
            verdict,
            pairwise_result.a_wins, pairwise_result.b_wins, pairwise_result.ties,
            pairwise_result.a_weighted, pairwise_result.b_weighted,
            pairwise_result.a_weighted_score * 100,
        )

    def _attach_match_critiques(
        self,
        *,
        result,
        pairwise_result,
        champion_id: str | None,
    ) -> tuple[Optional[object], Optional[object]]:
        """Persist the per-perspective critiques.

        - Challenger's critique goes into result.private_metrics so it
          persists with the new program via shinka's normal metrics.json
          → DB path.
        - Champion's critique is appended to its existing DB row via
          append_match_critique_threadsafe (opens its own connection).

        Returns the (challenger_critique, champion_critique) pair so callers
        can decide what to do downstream (e.g. brief refresh).
        """
        challenger_critique = pairwise_result.critiques_by_label.get("a")
        champion_critique = pairwise_result.critiques_by_label.get("b")
        if challenger_critique is not None:
            existing = result.private_metrics.setdefault("match_critiques", [])
            existing.append(challenger_critique.model_dump())
        if champion_critique is not None and champion_id and champion_id != "pending":
            self.db.append_match_critique_threadsafe(
                champion_id, champion_critique.model_dump()
            )
        return challenger_critique, champion_critique

    async def _evaluate_with_pairwise_async(
        self, program_path: str, results_dir: str, config_path: str,
        island_idx: int,
    ) -> None:
        """Async implementation of validate + pairwise. Linear sequence:
        validate → load champion → parse genomes → pairwise → apply outcome
        → attach critiques → refresh lineage briefs → write metrics.
        """
        from owtn.evaluation.stage_1 import evaluate

        result = await evaluate(program_path, results_dir, config_path)
        if not result.correct:
            return

        champion_data = self._read_champion_data(island_idx)
        if champion_data is None:
            result.combined_score = 0.5
            result.text_feedback = "No champion found — initial champion."
            self._write_eval_result(result, results_dir)
            logger.info("Initial champion (island %s, score=50%%)", island_idx)
            return

        try:
            challenger_genome = ConceptGenome.model_validate_json(
                Path(program_path).read_text()
            )
            champion_genome = ConceptGenome.model_validate_json(champion_data["code"])
        except Exception as e:
            logger.warning("Could not parse genomes for pairwise: %s", e)
            result.combined_score = 0.0
            result.text_feedback = f"Parse error: {e}"
            self._write_eval_result(result, results_dir)
            return

        logger.info(
            "'%s...' vs champion '%s...' — comparing...",
            challenger_genome.premise[:60], champion_genome.premise[:60],
        )

        try:
            pairwise_result = await pairwise_compare(
                genome_a=challenger_genome,
                genome_b=champion_genome,
                config=self.stage_config,
                champion_label="b",
            )
        except Exception as e:
            logger.error("Pairwise comparison failed: %s", e, exc_info=True)
            result.combined_score = 0.0
            result.text_feedback = f"Comparison failed: {e}"
            self._write_eval_result(result, results_dir)
            return

        self._apply_pairwise_outcome(
            result=result,
            pairwise_result=pairwise_result,
            champion_data=champion_data,
            program_path=program_path,
            island_idx=island_idx,
        )

        champion_id = champion_data.get("id")
        _, champion_critique = self._attach_match_critiques(
            result=result,
            pairwise_result=pairwise_result,
            champion_id=champion_id,
        )

        # Compute lineage briefs in-band, before metrics.json is written or
        # the champion's defense critique is read by any subsequent
        # mutation prompt. This eliminates the precompute race
        # (lab/issues/closed/2026-04-19-parent-brief-precompute-race.md) — by
        # the time shinka adds the challenger to DB or picks the champion as
        # a parent, both have fresh `lineage_brief_rendered` in their rows.
        await self._compute_briefs_in_band(
            result=result,
            challenger_genome=challenger_genome,
            champion_id=champion_id,
            champion_genome=champion_genome if champion_critique is not None else None,
        )

        self._write_eval_result(result, results_dir)

    async def _compute_briefs_in_band(
        self,
        *,
        result,
        challenger_genome: ConceptGenome,
        champion_id: str | None,
        champion_genome: ConceptGenome | None,
    ) -> None:
        """Refresh lineage briefs for the challenger (in-memory) and the
        champion (via DB threadsafe write). Called from the eval worker
        before metrics.json is written, so shinka's subsequent reads see
        fresh briefs without depending on the per-generation precompute.
        See lab/issues/closed/2026-04-19-parent-brief-precompute-race.md.
        """
        from owtn.optimizer.adapters import compute_stage_1_lineage_brief

        classifier_model = self.stage_config.llm.classifier_model

        # Challenger: brief reflects the single just-completed match,
        # already attached to result.private_metrics["match_critiques"].
        try:
            rendered, payload = await compute_stage_1_lineage_brief(
                self_genome=challenger_genome.model_dump(),
                private_metrics=result.private_metrics,
                classifier_model=classifier_model,
            )
            if payload is not None:
                result.private_metrics["lineage_brief_cache"] = payload
            result.private_metrics["lineage_brief_rendered"] = rendered
        except Exception as e:
            logger.warning("Challenger brief computation failed: %s", e)

        # Champion: re-read post-append critiques from DB, compute brief,
        # write back via threadsafe. Skipped if the champion is the synthetic
        # "pending" placeholder (i.e. mid-succession on a fresh upset).
        if (
            champion_id
            and champion_id != "pending"
            and champion_genome is not None
        ):
            try:
                champ_pm = self.db.get_program_private_metrics_threadsafe(
                    champion_id
                )
                rendered, payload = await compute_stage_1_lineage_brief(
                    self_genome=champion_genome.model_dump(),
                    private_metrics=champ_pm,
                    classifier_model=classifier_model,
                )
                if payload is not None:
                    self.db.set_lineage_brief_threadsafe(
                        champion_id, payload, rendered
                    )
            except Exception as e:
                logger.warning(
                    "Champion brief computation failed for %s: %s",
                    champion_id[:8], e,
                )

    @staticmethod
    def _write_eval_result(result, results_dir: str) -> None:
        """Re-write metrics.json with updated pairwise scores."""
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        (results_path / "metrics.json").write_text(result.model_dump_json(indent=2))
        (results_path / "correct.json").write_text(json.dumps({"correct": result.correct}))

    async def run_async(self):
        """Configure logging and snapshot gen 0, then run evolution."""
        # Context vars must be set inside the async context to survive asyncio.run().
        llm_log_dir.set(self._log_dir)
        llm_context.set({"role": "generation"})

        # Set initial annealing state (early phase).
        self._anneal(0)

        orig_setup = self._setup_async

        async def _setup_with_snapshot():
            await orig_setup()
            # Set gen 0 programs as initial champions (score 0.5).
            # They bypass the normal generation flow, so _run_pairwise_selection
            # never sees them.
            for program in self.db.get_programs_by_generation(0):
                if program.correct:
                    self.db.update_program_score(program.id, 0.5, "Initial champion.")
                    name = (program.metadata or {}).get("patch_name", program.id[:8])
                    logger.info("Gen 0: '%s' → initial champion (island %s, score=50%%)", name, program.island_idx)
            self._write_champions_to_disk()
            self._snapshot(0)

        self._setup_async = _setup_with_snapshot
        await super().run_async()

        # Post-evolution: Swiss tournament across island champions.
        await self._run_final_tournament()

        # End-of-run statistics report — judge diagnostics, cost breakdown,
        # evolution dynamics. Appended to evolution_run.log + stats.json.
        try:
            from owtn.stats_report import write_stats
            write_stats(Path(self.results_dir))
        except Exception as e:
            logger.warning("Failed to write stats report: %s", e)

    async def _update_completed_generations(self):
        """Snapshot state, write champions to disk, and anneal after each generation."""
        old = self.completed_generations
        await super()._update_completed_generations()
        if self.completed_generations > old:
            gen = self.completed_generations - 1
            self._write_champions_to_disk()
            await self._precompute_parent_briefs()
            # Population brief reads from the just-refreshed lineage briefs;
            # order is load-bearing (lineage first, then population).
            await self._compute_population_brief()
            self._snapshot(gen)
            self._anneal(gen)

    async def _compute_population_brief(self) -> None:
        """End-of-gen: regenerate the run-wide population brief from all
        cached lineage briefs. Stores two rendered blocks on the runner —
        `population_context` (goes in the middle of the mutation prompt) and
        `exploration_directions` (goes at top AND end of the user message,
        instruction-sandwich style for primacy + recency).

        Skipped entirely if `LLMConfig.run_brief_model` is not set.
        """
        model = self.stage_config.llm.run_brief_model
        if not model:
            return
        from owtn.optimizer.adapters import compute_stage_1_population_brief

        try:
            result = await compute_stage_1_population_brief(
                db=self.db,
                run_brief_model=model,
                judge_names=list(self.stage_config.judges.panel),
            )
        except Exception as e:
            logger.warning("Population brief computation failed: %s", e)
            return
        if result is not None:
            context, directions = result
            self._latest_population_context = context
            self._latest_exploration_directions = directions
            # PromptSampler.sample() reads these attributes when building
            # each mutation's user message.
            self.prompt_sampler.population_context = context
            self.prompt_sampler.exploration_directions = directions

    async def _precompute_parent_briefs(self) -> None:
        """Refresh LineageBrief for any program whose match_critiques has grown.

        Called post-generation, before the next generation's mutation phase.
        Iterates *all* correct programs (not just current island champions)
        and recomputes a brief for any whose cached count is stale relative
        to its accumulated `match_critiques` length. This catches deposed
        champions that continued accumulating defense critiques during the
        generation in which they were dethroned — those concepts still get
        sampled later as inspirations or by the bandit's parent selector,
        and their briefs need to reflect the full critique history.

        Uses lazy caching keyed on `len(match_critiques)`: programs whose
        critique count hasn't grown since the last brief are skipped.
        """
        from owtn.optimizer.adapters import compute_stage_1_lineage_brief
        import json

        classifier_model = self.stage_config.llm.classifier_model

        self.db.cursor.execute(
            "SELECT id, code, private_metrics FROM programs WHERE correct = 1"
        )
        rows = self.db.cursor.fetchall()

        for row in rows:
            program_id = row["id"]
            code = row["code"]
            try:
                pm = json.loads(row["private_metrics"] or "{}")
            except json.JSONDecodeError:
                pm = {}
            critique_count = len(pm.get("match_critiques") or [])
            if critique_count == 0:
                continue
            cached_count = (pm.get("lineage_brief_cache") or {}).get("count")
            if cached_count == critique_count:
                continue  # cache fresh — skip

            try:
                self_genome = json.loads(code)
            except Exception as e:
                logger.debug(
                    "Skipping brief for %s — genome parse failed: %s",
                    program_id[:8], e,
                )
                continue
            try:
                rendered, payload = await compute_stage_1_lineage_brief(
                    self_genome=self_genome,
                    private_metrics=pm,
                    classifier_model=classifier_model,
                )
            except Exception as e:
                logger.warning(
                    "Brief precompute failed for %s: %s", program_id[:8], e,
                )
                continue
            if payload is None and pm.get("lineage_brief_rendered") == rendered:
                continue  # nothing to write
            new_pm = dict(pm)
            if payload is not None:
                new_pm["lineage_brief_cache"] = payload
            new_pm["lineage_brief_rendered"] = rendered
            try:
                self.db.cursor.execute(
                    "UPDATE programs SET private_metrics = ? WHERE id = ?",
                    (json.dumps(new_pm), program_id),
                )
                self.db.conn.commit()
            except Exception as e:
                logger.warning(
                    "Failed to persist brief for %s: %s", program_id[:8], e,
                )


    def _log_evolution_summary(self) -> None:
        """Log a clear island-by-island evolution summary."""
        num_islands = self.stage_config.database.num_islands
        total_gens = self.stage_config.evolution.num_generations

        logger.info("")
        logger.info("=" * 60)
        logger.info("  EVOLUTION SUMMARY")
        logger.info("=" * 60)

        for island_idx in range(num_islands):
            programs = []
            for gen in range(total_gens):
                for p in self.db.get_programs_by_generation(gen):
                    if p.island_idx == island_idx and p.correct:
                        programs.append(p)

            champion = self.db.get_island_champion(island_idx)
            champ_name = "none"
            if champion:
                try:
                    g = ConceptGenome.model_validate_json(champion.code)
                    champ_name = g.premise[:55]
                except Exception:
                    champ_name = (champion.metadata or {}).get("patch_name", champion.id[:8])

            logger.info("")
            logger.info("  Island %d  (%d concepts, champion: '%s...')", island_idx, len(programs), champ_name)
            logger.info("  " + "-" * 56)

            for p in sorted(programs, key=lambda x: x.generation):
                try:
                    g = ConceptGenome.model_validate_json(p.code)
                    premise = g.premise[:50]
                except Exception:
                    premise = (p.metadata or {}).get("patch_name", p.id[:8])

                is_champ = " ★" if champion and p.id == champion.id else ""
                score_pct = p.combined_score * 100
                logger.info(
                    "    gen %2d  %.0f%%  '%s...'%s",
                    p.generation, score_pct, premise, is_champ,
                )

        logger.info("")
        logger.info("=" * 60)

    async def _run_final_tournament(self) -> None:
        """Swiss tournament across island champions after evolution completes."""
        self._log_evolution_summary()

        num_islands = self.stage_config.database.num_islands
        participants = []

        for island_idx in range(num_islands):
            champion = self.db.get_island_champion(island_idx)
            if champion is None:
                continue
            try:
                genome = ConceptGenome.model_validate_json(champion.code)
                participants.append((champion.id, genome))
            except Exception as e:
                logger.warning("Could not parse champion for island %d: %s", island_idx, e)

        if len(participants) < 2:
            logger.info("Tournament skipped: fewer than 2 island champions.")
            return

        logger.info("")
        logger.info("Running final Swiss tournament with %d island champions...", len(participants))
        llm_context.set({"role": "tournament"})

        rankings = await run_tournament(participants, self.stage_config)

        # Write tournament results.
        results_path = Path(self.results_dir) / "tournament.json"
        import json
        results_path.write_text(json.dumps([
            {
                "rank": rank,
                "program_id": entry.program_id,
                "wins": entry.wins,
                "losses": entry.losses,
                "buchholz": entry.buchholz,
                "matches": entry.match_history,
            }
            for rank, entry in enumerate(rankings, 1)
        ], indent=2))
        logger.info("Tournament results written to %s", results_path)

        # Apply tournament bonus: score = (tournament_wins + 0.5) / (total_rounds + 1)
        # This spreads champions across the 0-1 range by tournament performance.
        n = len(rankings)
        total_rounds = rankings[0].wins + rankings[0].losses if rankings else 1

        logger.info("")
        logger.info("=" * 60)
        logger.info("  TOURNAMENT RESULTS")
        logger.info("=" * 60)
        for rank, entry in enumerate(rankings, 1):
            score = (entry.wins + 0.5) / (total_rounds + 1)
            self.db.update_program_score(entry.program_id, score, "")
            premise = entry.genome.premise[:50]
            logger.info(
                "  #%d  W%d-L%d  score=%.0f%%  '%s...'",
                rank, entry.wins, entry.losses, score * 100, premise,
            )
            for match in entry.match_history:
                if match.get("opponent") == "bye":
                    logger.info("       bye")
                    continue
                dims = match.get("dimension_wins", {})
                won = [d for d, w in dims.items() if w == ("a" if match["result"] == "win" else "b")]
                lost = [d for d, w in dims.items() if w == ("b" if match["result"] == "win" else "a")]
                logger.info(
                    "       %s %s  (won: %s | lost: %s)",
                    match["result"].upper(), match.get("score", "?"),
                    ", ".join(won) or "none",
                    ", ".join(lost) or "none",
                )
        logger.info("=" * 60)

        if rankings:
            best = rankings[0]
            logger.info("Winner: %s", best.program_id[:8])
            best_dir = Path(self.results_dir) / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            (best_dir / "main.json").write_text(
                best.genome.model_dump_json(indent=2)
            )

    def _anneal(self, generation: int) -> None:
        """Anneal genesis ratio: high early, lower late.

        Temperature is fixed per-model (`GenerationModelConfig.temperature`),
        not annealed.
        """
        sched = self.stage_config.evolution.annealing
        total = self.stage_config.evolution.num_generations
        warmup_gens = int(total * sched.warmup_fraction)
        is_early = generation < warmup_gens
        self.prompt_sampler.genesis_ratio = (
            sched.genesis_ratio_early if is_early else sched.genesis_ratio_late
        )
        logger.debug(
            "Generation %d: genesis_ratio=%.2f",
            generation, self.prompt_sampler.genesis_ratio,
        )

    def _snapshot(self, generation: int) -> None:
        try:
            snapshot_generation(
                db=self.db,
                generation=generation,
                results_dir=self.results_dir,
                total_api_cost=self.total_api_cost,
            )
        except Exception as e:
            logger.debug("State snapshot failed: %s", e)

    async def _generate_one_concept(self) -> tuple[str, str, str, float, dict]:
        """Generate a single concept genome with fresh random rolls.

        Returns (code, patch_name, patch_description, total_costs, llm_metadata).
        """
        genesis_types, genesis_probs = filter_genesis_eligible(
            self.stage_config.evolution.patch_types,
            self.stage_config.evolution.patch_type_probs,
        )
        operator = str(np.random.choice(genesis_types, p=genesis_probs))
        tonal_text, register_name, mode_name = sample_tonal_steering()
        llm_context.set({"role": "generation", "operator": operator, "generation": 0})

        sys_msg, user_msg = build_operator_prompt(
            operator,
            registry=self.registry,
            is_initial=True,
            seed_bank=self.seed_bank,
            used_seed_ids=self.used_seed_ids,
            sampling_strategy=self.stage_config.seed_sampling_strategy,
            prompt=self.stage_config.prompt,
            tonal_steering=tonal_text,
        )

        model_sample_probs = None
        model_posterior = None
        if self.llm_selection is not None:
            model_sample_probs, model_posterior = self.llm_selection.select_llm()
        else:
            # Build sampling probs from per-model weights when present.
            gm = self.stage_config.llm.generation_models
            weights = [m.weight for m in gm]
            total = sum(weights)
            if total > 0:
                model_sample_probs = [w / total for w in weights]

        llm_kwargs = self.llm.get_kwargs(model_sample_probs=model_sample_probs)
        total_costs = 0.0

        for attempt in range(self.evo_config.max_patch_attempts):
            response = await self.llm.query(
                msg=user_msg,
                system_msg=sys_msg,
                model_sample_probs=model_sample_probs,
                model_posterior=model_posterior,
            )

            if response is None or response.content is None:
                error_msg = "LLM response content was None."
                logger.info(
                    f"  INITIAL [{operator}] ATTEMPT {attempt + 1}/"
                    f"{self.evo_config.max_patch_attempts} FAILURE: {error_msg}"
                )
                await self._save_patch_attempt_async(
                    generation=0, novelty_attempt=1, resample_attempt=1,
                    patch_attempt=attempt + 1, response=response,
                    error_msg=error_msg, patch_text=None, num_applied=0,
                    patch_name=None, patch_description=None, success=False,
                )
                if attempt < self.evo_config.max_patch_attempts - 1:
                    user_msg = (
                        "The previous response was empty. Please try again "
                        "and provide the full concept genome as JSON."
                    )
                    continue
                else:
                    break

            total_costs += response.cost or 0.0

            initial_code = extract_between(
                response.content, "```json", "```", False,
            )

            if initial_code:
                patch_name = extract_between(
                    response.content, "<NAME>", "</NAME>", False,
                )
                patch_description = extract_between(
                    response.content, "<DESCRIPTION>", "</DESCRIPTION>", False,
                )

                logger.info(
                    f"  INITIAL [{operator}] ATTEMPT {attempt + 1}/"
                    f"{self.evo_config.max_patch_attempts} SUCCESS."
                )

                await self._save_patch_attempt_async(
                    generation=0, novelty_attempt=1, resample_attempt=1,
                    patch_attempt=attempt + 1, response=response,
                    error_msg=None, patch_text=initial_code, num_applied=1,
                    patch_name=patch_name, patch_description=patch_description,
                    success=True,
                )

                llm_metadata = {
                    "patch_type": operator,
                    "affective_register": register_name,
                    "literary_mode": mode_name,
                    "api_costs": total_costs,
                    "num_applied": 1,
                    "patch_name": patch_name,
                    "patch_description": patch_description,
                    "error_attempt": None,
                    "novelty_attempt": 1,
                    "resample_attempt": 1,
                    "patch_attempt": attempt + 1,
                    **llm_kwargs,
                    "llm_result": response.to_dict() if response else None,
                    "diff_summary": {},
                }

                return initial_code, patch_name, patch_description, total_costs, llm_metadata
            else:
                error_msg = "Could not extract JSON from response."
                logger.info(
                    f"  INITIAL [{operator}] ATTEMPT {attempt + 1}/"
                    f"{self.evo_config.max_patch_attempts} FAILURE: {error_msg}"
                )
                await self._save_patch_attempt_async(
                    generation=0, novelty_attempt=1, resample_attempt=1,
                    patch_attempt=attempt + 1, response=response,
                    error_msg=error_msg, patch_text=None, num_applied=0,
                    patch_name=None, patch_description=None, success=False,
                )
                if attempt < self.evo_config.max_patch_attempts - 1:
                    user_msg = (
                        "Could not extract code from your last response. "
                        "Please enclose the concept genome in ```json...``` tags."
                    )
                else:
                    break

        raise RuntimeError(
            f"LLM failed to generate a valid initial concept after "
            f"{self.evo_config.max_patch_attempts} attempts."
        )

    async def _generate_initial_program(self):
        """Cold-start: generate a unique concept for each island."""
        import asyncio

        num_islands = self.stage_config.database.num_islands

        # Generate unique concepts — one per island, each with fresh random rolls.
        concepts = []
        for i in range(num_islands):
            logger.info(f"  Generating initial concept for island {i+1}/{num_islands}...")
            concepts.append(await self._generate_one_concept())

        # Island 0: use the standard setup path (creates Gen 0 dir, evals, etc.)
        code, name, desc, cost, meta = concepts[0]
        await self._setup_initial_program_with_metadata(code, name, desc, cost, meta)

        if num_islands <= 1:
            return

        # The standard path copied island 0's concept to all other islands.
        # Delete those copies — we have unique concepts for each.
        self.db.cursor.execute(
            "DELETE FROM programs WHERE json_extract(metadata, '$._is_island_copy') = 1"
        )
        self.db.cursor.execute(
            "DELETE FROM map_elites_cells WHERE program_id NOT IN (SELECT id FROM programs)"
        )
        self.db.conn.commit()
        logger.info(
            f"  Deleted island copies; generating unique concepts for "
            f"islands 1-{num_islands - 1}."
        )

        # Islands 1+: evaluate each unique concept and add to DB directly.
        # Each concept gets its own subdir in Gen 0 for proper audit trail.
        gen_dir = f"{self.results_dir}/{FOLDER_PREFIX}_0"

        for island_idx in range(1, num_islands):
            code, name, desc, cost, meta = concepts[island_idx]

            # Write to island-specific file so we don't overwrite island 0's main.json.
            island_dir = f"{gen_dir}/island_{island_idx}"
            island_results = f"{island_dir}/results"
            Path(island_dir).mkdir(parents=True, exist_ok=True)
            Path(island_results).mkdir(parents=True, exist_ok=True)

            exec_fname = f"{island_dir}/main.{self.lang_ext}"
            await write_file_async(exec_fname, code)

            # Evaluate.
            loop = asyncio.get_event_loop()
            try:
                results, rtime = await loop.run_in_executor(
                    None, self.scheduler.run, exec_fname, island_results,
                )
                logger.info(
                    f"  Island {island_idx} concept evaluated in {rtime:.2f}s"
                )
            except Exception as e:
                logger.warning(f"  Island {island_idx} evaluation failed: {e}")
                results = {}

            # Extract metrics.
            correct_val = results.get("correct", {}).get("correct", False)
            metrics_val = results.get("metrics", {})

            # Compute embedding.
            code_embedding, e_cost = await self._get_code_embedding_async(exec_fname)

            # Build program.
            meta["embed_cost"] = e_cost
            meta["novelty_cost"] = 0.0
            program = Program(
                id=str(uuid.uuid4()),
                code=code,
                generation=0,
                correct=correct_val,
                combined_score=metrics_val.get("combined_score", 0.0),
                public_metrics=metrics_val.get("public_metrics", {}),
                private_metrics=metrics_val.get("private_metrics", {}),
                text_feedback=metrics_val.get("text_feedback", ""),
                timestamp=datetime.now().timestamp(),
                embedding=code_embedding,
                metadata=meta,
            )

            # Add to DB — island manager routes to next uninitialized island.
            await self.async_db.add_program_async(program)
            self.total_api_cost += cost + e_cost
            logger.info(
                f"  Island {island_idx} seeded: {name or 'unnamed'} "
                f"(correct={correct_val}, score={program.combined_score:.3f})"
            )
