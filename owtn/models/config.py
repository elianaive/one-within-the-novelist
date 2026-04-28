from __future__ import annotations

from pydantic import BaseModel, field_validator

from owtn.llm.kwargs import THINKING_TOKENS


_REASONING_EFFORTS = {"disabled", *THINKING_TOKENS.keys()}


class GenerationModelConfig(BaseModel):
    """Per-model generation params. Each entry in `LLMConfig.generation_models`
    has its own sampling params; the pipeline picks one model per generation
    call using `weight` (uniform when all are equal), then uses THAT model's
    params for the call. Lets different-family models (e.g. deepseek-v4-pro +
    claude-sonnet-4-6) run with their own optimal temperatures / reasoning
    efforts without forcing a single global value.
    """

    name: str
    weight: float = 1.0
    temperature: float = 1.0
    reasoning_effort: str = "disabled"
    # Explicit thinking-token budget for Anthropic / Gemini reasoning models.
    # Preferred over the discrete `reasoning_effort` enum when you want a
    # specific budget (e.g. 6000 instead of "high"=8192). When None, the
    # provider falls back to the THINKING_TOKENS[reasoning_effort] mapping
    # — so existing configs without thinking_tokens keep behaving the same.
    # Has no effect on OpenAI/DeepSeek models, which take an effort string.
    thinking_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    # When true, every generation/genesis/mutation call routed to this model
    # goes through a critique-revise cycle: initial output → same-model critic
    # (given the critic system prompt) → revised output by the original
    # generator conditioned on the critic's response. Adds ~2x generation
    # cost. See lab/issues/2026-04-24-self-critic-critique-revise.md.
    self_critic: bool = False
    # Reasoning effort for the *critic* sub-call (the review of the genome).
    # Defaults to "disabled" — critique IS the thinking, so internal
    # reasoning is redundant on this call. The revise sub-call always
    # inherits the generator's reasoning_effort, since it's doing the real
    # work of integrating the critique into a revised genome and benefits
    # from extended reasoning. Accepts the same values as reasoning_effort;
    # only "disabled" is actively supported today, other values pass kwargs
    # through unchanged.
    self_critic_reasoning_effort: str = "disabled"

    @field_validator("reasoning_effort", "self_critic_reasoning_effort")
    @classmethod
    def _check_effort(cls, v: str) -> str:
        if v not in _REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of "
                f"{sorted(_REASONING_EFFORTS)}, got {v!r}"
            )
        return v

    @field_validator("weight")
    @classmethod
    def _check_weight(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"weight must be > 0, got {v!r}")
        return v


class LLMConfig(BaseModel):
    generation_models: list[GenerationModelConfig]
    classifier_model: str
    embedding_model: str
    # RunBrief summarizer. When None, the end-of-gen population-brief pass
    # is skipped entirely and mutation prompts receive no population signal.
    # See `lab/issues/2026-04-22-global-optimizer-state.md`.
    run_brief_model: str | None = None
