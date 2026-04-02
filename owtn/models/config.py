from __future__ import annotations

from pydantic import BaseModel


class LLMConfig(BaseModel):
    generation_models: list[str]
    generation_model_family: str
    judge_models: list[str]
    classifier_model: str
    embedding_model: str
