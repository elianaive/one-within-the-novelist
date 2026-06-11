"""Stage 4 manuscript representation.

The canonical state of the prose lives on disk at `story.md`. These
Pydantic models are the parsed view of that file — they are derived from
the file, not the source of truth. The file always wins; models are
recomputed by re-parsing.

File format: optional YAML frontmatter (`---` ... `---`), then markdown
where each scene is a level-2 heading whose text is the scene_id matching
a Stage 2 DAG node id. Convention only; the agent is free to restructure.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Scene(BaseModel):
    """One parsed scene from `story.md`.

    `body` is the text between this scene's heading and the next one
    (or end of file), stripped of leading/trailing blank lines. An empty
    body means the scaffolding is in place but no prose has been written
    yet — common during DownDraft before the agent reaches that scene.
    """
    id: str = Field(min_length=1)
    body: str = ""

    @property
    def word_count(self) -> int:
        return len(self.body.split())

    @property
    def is_empty(self) -> bool:
        return not self.body.strip()


class Manuscript(BaseModel):
    """Parsed `story.md`.

    `frontmatter` carries any YAML the file opens with — run id, generator
    model, version stamps. None when the file has no frontmatter block.
    `scenes` is the in-file order, which during DownDraft tracks the DAG's
    topological order but is not enforced.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frontmatter: dict | None = None
    scenes: list[Scene] = Field(default_factory=list)

    def scene_ids(self) -> list[str]:
        return [s.id for s in self.scenes]

    def find(self, scene_id: str) -> Scene | None:
        for s in self.scenes:
            if s.id == scene_id:
                return s
        return None

    @property
    def total_word_count(self) -> int:
        return sum(s.word_count for s in self.scenes)
