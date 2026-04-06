"""Stage 1 concept evolution runner — thin wrapper around ShinkaEvolveRunner.

Configures the loop for JSON concept genomes, dispatches through the operator
registry, and overrides initial population generation with cold-start allocation.
"""

from __future__ import annotations

import json
import logging
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
from owtn.prompts.stage_1.registry import build_operator_prompt, load_registry
from owtn.state_logger import snapshot_generation

logger = logging.getLogger(__name__)

# Cold-start operator weights from docs/stage-1/population.md.
# compost and crossover excluded — they need existing population.
COLD_START_OPERATORS = {
    "collision": 0.20,
    "thought_experiment": 0.20,
    "noun_list": 0.15,
    "constraint_first": 0.15,
    "anti_premise": 0.10,
    "discovery": 0.10,
    "compression": 0.05,
    "real_world_seed": 0.05,
}
_CS_NAMES = list(COLD_START_OPERATORS.keys())
_CS_PROBS = np.array(list(COLD_START_OPERATORS.values()))
_CS_PROBS = _CS_PROBS / _CS_PROBS.sum()  # ensure exact 1.0


def _build_shinka_configs(
    cfg: StageConfig,
    config_path: str,
) -> tuple[ShinkaEvolutionConfig, ShinkaDatabaseConfig, LocalJobConfig]:
    """Translate our StageConfig into ShinkaEvolve dataclass configs."""
    evo = ShinkaEvolutionConfig(
        task_sys_msg=None,  # set by PromptSampler via registry
        language=cfg.evolution.language,
        patch_types=cfg.evolution.patch_types,
        patch_type_probs=cfg.evolution.patch_type_probs,
        num_generations=cfg.evolution.num_generations,
        max_patch_resamples=cfg.evolution.max_patch_resamples,
        llm_models=cfg.llm.generation_models,
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
        )
        self.prompt_sampler.tonal_inherit_rate = self.stage_config.evolution.tonal_inherit_rate
        self.prompt_sampler.tonal_crossover_new_rate = self.stage_config.evolution.tonal_crossover_new_rate

        # Set eval_function now that self.db is available (from super().__init__).
        # The eval function reads champion from a file (not DB) to avoid
        # SQLite threading issues — the scheduler runs it in a thread pool.
        self._champions_dir = Path(self.results_dir) / "champions"
        self._champions_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler.config.eval_function = self._evaluate_with_pairwise

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
            }))

    def _evaluate_with_pairwise(
        self, program_path: str, results_dir: str, config_path: str,
        parent_id: str | None = None, island_idx: int | None = None,
    ) -> None:
        """Inline evaluation: validate + pairwise in one blocking step.

        Called by ShinkaEvolve's scheduler in a worker thread. Validates
        the genome, reads the champion from disk (written by main thread),
        runs pairwise comparison, and writes metrics.json. Blocks until
        pairwise completes, so ShinkaEvolve waits for the real score.
        """
        import asyncio
        from owtn.evaluation.stage_1 import evaluate

        # Propagate context vars to this worker thread (they don't cross
        # thread boundaries automatically).
        llm_log_dir.set(self._log_dir)
        llm_context.set({"role": "pairwise_judge"})

        # Step 1: Validate.
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(evaluate(program_path, results_dir, config_path))

        if not result.correct:
            loop.close()
            return

        # Step 2: Read island champion from disk.
        champion_data = None
        if island_idx is not None:
            champ_file = self._champions_dir / f"island_{island_idx}.json"
            if champ_file.exists():
                try:
                    champion_data = json.loads(champ_file.read_text())
                except Exception:
                    pass

        if champion_data is None:
            result.combined_score = 0.5
            result.text_feedback = "No champion found — initial champion."
            self._write_eval_result(result, results_dir)
            logger.info("'%s' → initial champion (score=50%%)", Path(program_path).parent.name)
            loop.close()
            return

        # Step 3: Pairwise comparison.
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
            loop.close()
            return

        challenger_name = Path(program_path).parent.name
        champion_name = (champion_data.get("metadata") or {}).get("patch_name", champion_data["id"][:8])
        logger.info("'%s' vs champion '%s' — comparing...", challenger_name, champion_name)

        try:
            pairwise_result = loop.run_until_complete(pairwise_compare(
                genome_a=challenger_genome,
                genome_b=champion_genome,
                config=self.stage_config,
                champion_label="b",
            ))
        except Exception as e:
            logger.error("Pairwise comparison failed: %s", e, exc_info=True)
            result.combined_score = 0.0
            result.text_feedback = f"Comparison failed: {e}"
            self._write_eval_result(result, results_dir)
            loop.close()
            return
        finally:
            loop.close()

        if pairwise_result.winner == "a":
            result.combined_score = pairwise_result.a_score
            result.text_feedback = pairwise_result.feedback
            logger.info(
                "'%s' BEATS '%s' (%d-%d-%d, score=%.0f%%)",
                challenger_name, champion_name,
                pairwise_result.a_wins, pairwise_result.b_wins, pairwise_result.ties,
                pairwise_result.a_score * 100,
            )
        else:
            result.combined_score = pairwise_result.a_score * 0.9
            result.text_feedback = pairwise_result.feedback
            logger.info(
                "'%s' loses to '%s' (%d-%d-%d, score=%.0f%%)",
                challenger_name, champion_name,
                pairwise_result.a_wins, pairwise_result.b_wins, pairwise_result.ties,
                pairwise_result.a_score * 100,
            )

        self._write_eval_result(result, results_dir)

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

    async def _update_completed_generations(self):
        """Snapshot state, write champions to disk, and anneal after each generation."""
        old = self.completed_generations
        await super()._update_completed_generations()
        if self.completed_generations > old:
            gen = self.completed_generations - 1
            self._write_champions_to_disk()
            self._snapshot(gen)
            self._anneal(gen)


    async def _run_final_tournament(self) -> None:
        """Swiss tournament across island champions after evolution completes."""
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
        logger.info("=" * 50)
        logger.info("  FINAL TOURNAMENT RANKINGS")
        logger.info("=" * 50)
        for rank, entry in enumerate(rankings, 1):
            score = (entry.wins + 0.5) / (total_rounds + 1)
            self.db.update_program_score(entry.program_id, score, "")
            logger.info(
                "  #%d  %-12s  W%d-L%d  Buchholz=%d  score=%.0f%%",
                rank, entry.program_id[:8], entry.wins, entry.losses,
                entry.buchholz, score * 100,
            )
        logger.info("=" * 50)

        if rankings:
            best = rankings[0]
            logger.info("Winner: %s", best.program_id[:8])
            best_dir = Path(self.results_dir) / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            (best_dir / "main.json").write_text(
                best.genome.model_dump_json(indent=2)
            )

    def _anneal(self, generation: int) -> None:
        """Anneal temperature and genesis ratio: high early, lower late."""
        sched = self.stage_config.evolution.annealing
        total = self.stage_config.evolution.num_generations
        warmup_gens = int(total * sched.warmup_fraction)
        is_early = generation < warmup_gens
        self.llm.temperatures = sched.temp_early if is_early else sched.temp_late
        self.prompt_sampler.genesis_ratio = (
            sched.genesis_ratio_early if is_early else sched.genesis_ratio_late
        )
        logger.debug(
            "Generation %d: temps=%s, genesis_ratio=%.2f",
            generation, self.llm.temperatures, self.prompt_sampler.genesis_ratio,
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
        operator = str(np.random.choice(_CS_NAMES, p=_CS_PROBS))
        tonal_text, register_name, mode_name = sample_tonal_steering()
        llm_context.set({"role": "generation", "operator": operator, "generation": 0})

        sys_msg, user_msg = build_operator_prompt(
            operator,
            registry=self.registry,
            is_initial=True,
            seed_bank=self.seed_bank,
            steering=self.stage_config.steering,
            tonal_steering=tonal_text,
        )

        model_sample_probs = None
        model_posterior = None
        if self.llm_selection is not None:
            model_sample_probs, model_posterior = self.llm_selection.select_llm()

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
