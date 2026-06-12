# One Within the Novelist

<p align="center"><img src="docs/assets/one-within-the-villainess.webp" width="520" alt="A garden scene from The One Within the Villainess"></p>
<p align="center"><sub><em>The name riffs on <a href="https://en.wikipedia.org/wiki/Akuyaku_Reij%C5%8D_no_Naka_no_Hito">The One Within the Villainess</a>, a story about writing against a scripted ending from inside it. We take the same wager here — every ending pulls, and an author should hold more than one soul.</em></sub></p>

An evolutionary pipeline that writes short stories.

Two convictions drive the design. LLMs draft linearly while writers branch, discard, and double back — so every stage here is population-based search, not a single forward pass. And generating passable fiction is far easier than recognizing good fiction — so selection never asks a judge for a score. Judges see two candidates and pick a winner per criterion, in both orderings, on a panel drawn from different model families than the generator. Scores compress everything into the same narrow band; comparisons discriminate.

## Stages

Each stage evolves a different genome under judge pressure; all four run end-to-end. Full genome/operator/evaluation specs in [`docs/stages.md`](docs/stages.md).

| | Stage | What evolves |
|---|---|---|
| 1 | **Concept** | Premise genomes evolved on islands by a [domain modified ShinkaEvolve](lib/shinka-evolve). Challengers fight the island champion across 9 dimensions; a Swiss tournament ranks the survivors. |
| 2 | **Structure** | Typed-edge DAGs of story beats, grown by MCTS per concept. A final check verdicts each of the concept's structural demands. |
| 3 | **Voice** | Style genomes from a multi-agent session with deliberately constrained communication: agents draft in isolation, meet only through structured rounds of critique, revise with metric tools, and settle it by Borda vote. |
| 4 | **Prose** | The manuscript, in an RLM (recursive language model) shape: it lives on disk rather than in context, and edits dispatch to bounded subagents. One writer agent (PreThink → draft → revise) against a 15-critic ensemble plus on-demand domain experts, until issues plateau. |

## Running

API keys live in `.env` at the repo root (`ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `EXA_API_KEY`); `uv run` loads it automatically.

```bash
uv sync

# full pipeline, Stages 1→4, one results/run_<ts>/ root
uv run python -m owtn --pipeline-config configs/pipeline/dry_run.yaml

# or per stage
uv run python -m owtn.stage_1 --config configs/stage_1/light.yaml
uv run python -m owtn.stage_2 --config configs/stage_2/light.yaml --stage-1-results results/run_<ts>/stage_1/
```

Configs come in tiers — `dry_run` (mocked/cheap), `light`, `submission` — with cost estimates in each file's header.
