"""Stage 1 concept evolution runner — thin wrapper around ShinkaEvolveRunner.

Configures the loop for JSON concept genomes, dispatches through the operator
registry, and overrides initial population generation with cold-start allocation.
"""

from __future__ import annotations

import logging
import time
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
from shinka.launch import JobConfig, LocalJobConfig
from shinka.llm import extract_between
from shinka.utils import get_language_extension

from owtn.llm.call_logger import llm_context, llm_log_dir
from owtn.models.stage_1.config import StageConfig
from owtn.models.stage_1.seed_bank import SeedBank
from owtn.prompts.stage_1.registry import TONAL_TARGETS, build_operator_prompt, load_registry
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
        enable_dynamic_islands=cfg.database.enable_dynamic_islands,
        stagnation_threshold=cfg.database.stagnation_threshold,
        parent_selection_strategy=cfg.database.parent_selection_strategy,
    )

    job = LocalJobConfig(
        eval_program_path=str(
            Path(__file__).resolve().parent / "evaluation" / "__main__.py"
        ),
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
            self._snapshot(0)

        self._setup_async = _setup_with_snapshot
        await super().run_async()

    async def _update_completed_generations(self):
        """Snapshot state and update temperature schedule after each generation."""
        old = self.completed_generations
        await super()._update_completed_generations()
        if self.completed_generations > old:
            self._snapshot(self.completed_generations)
            self._anneal(self.completed_generations)

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

    async def _generate_initial_program(self):
        """Cold-start: generate a concept genome using operator allocation."""
        operator = str(np.random.choice(_CS_NAMES, p=_CS_PROBS))
        tonal_target = str(np.random.choice(TONAL_TARGETS))
        llm_context.set({"role": "generation", "operator": operator, "generation": 0})

        sys_msg, user_msg = build_operator_prompt(
            operator,
            registry=self.registry,
            is_initial=True,
            seed_bank=self.seed_bank,
            steering=self.stage_config.steering,
            tonal_steering=tonal_target,
        )

        # Select LLM
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

            # Extract JSON from ```json ... ``` code fence
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

                # No EVOLVE-BLOCK wrapping — JSON has no valid comment syntax.
                # apply_full.py already handles JSON without markers.

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

                await self._setup_initial_program_with_metadata(
                    initial_code,
                    patch_name,
                    patch_description,
                    total_costs,
                    llm_metadata,
                )
                return
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
