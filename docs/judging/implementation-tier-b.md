# Judging Implementation: Tier B LLM Judge Panel

The Tier B module orchestrates LLM-based evaluation — persona construction,
pointwise scoring, pairwise tournaments, dynamic rubrics, and quality gates.
This is where the real selection pressure lives.

---

## 1. Judge Persona Prompt Construction

The judge schema (defined in `docs/judging/overview.md`, implemented in
`configs/judges/*.yaml`) is formatted into placeholders in the judge system
prompt template (`owtn/prompts/stage_1/judge_system.txt`).

The template defines the full prompt structure including persona, reasoning
chain, and rubric anchors. The Python code below populates the template's
placeholders from the YAML config fields.

### Field Formatting

**Name and identity:**
```python
def format_judge_fields(judge: dict) -> dict:
    """Format YAML judge config into template placeholders."""
    return {
        "judge_name": judge["name"],
        "judge_identity": judge["identity"],
        "judge_values": "\n".join(f"{i+1}. {v}" for i, v in enumerate(judge["values"])),
        "judge_exemplars": "\n".join(f"- {ex}" for ex in judge.get("exemplars", [])),
        "judge_harshness": judge["harshness"],
        "harshness_instruction": HARSHNESS_INSTRUCTIONS[judge["harshness"]],
    }
```

**Harshness instructions:**
```python
HARSHNESS_INSTRUCTIONS = {
    "lenient": (
        "You focus on what works. You give generous interpretations of "
        "ambiguity. A 3/5 from you means significant problems. You believe "
        "in encouraging potential."
    ),
    "moderate": (
        "You identify both strengths and weaknesses with equal attention. "
        "Your scores reflect a balanced assessment. A 3/5 means competent "
        "but unremarkable."
    ),
    "demanding": (
        "You focus on what doesn't work. Your bar is high. A 4/5 from you "
        "is exceptional praise. You believe that honest criticism serves "
        "writers better than encouragement. You are not cruel, but you are "
        "exacting."
    ),
}
```

### Model Assignment

Each judge specifies which LLM family runs it. The critical rule: the judge's
model family MUST differ from the generation model family. Self-preference
bias (+0.52 on own outputs) is structurally eliminated by model diversity.

```python
def assign_models(judges: list[dict], generation_model: str,
                  available_models: list[str]) -> list[dict]:
    """Assign models to judges, excluding the generation model family."""
    non_gen_models = [m for m in available_models
                      if get_model_family(m) != get_model_family(generation_model)]
    for i, judge in enumerate(judges):
        judge["model"] = non_gen_models[i % len(non_gen_models)]
    return judges
```

---

## 2. Pointwise Evaluation Flow

Pointwise evaluation produces per-dimension scores for a single piece of
content. Used during evolutionary generations (every evaluation).

### Evaluation Prompt Template

The user message sent to each judge:

```
EVALUATE THE FOLLOWING:

---
{content}
---

SCORING RUBRIC:

For each dimension below, provide your reasoning (2-4 sentences referencing
specific elements) followed by your score (1-5).

{rubric_dimensions}

RESPOND IN THIS EXACT FORMAT:

## [Dimension Name]
**Reasoning:** [Your specific analysis, referencing concrete elements]
**Score:** [1-5]

## [Next Dimension]
...
```

### Rubric Dimension Formatting

For each dimension, include the name, description, and the full 1-5 anchors:

```python
def format_rubric_dimension(name: str, description: str,
                            anchors: dict[int, str]) -> str:
    section = f"### {name}\n{description}\n\n"
    section += "Score anchors:\n"
    for score in [1, 2, 3, 4, 5]:
        section += f"- **{score}/5:** {anchors[score]}\n"
    return section
```

### Response Parsing

```python
import re

def parse_pointwise_response(response: str, dimensions: list[str]) -> dict:
    """Parse structured judge response into scores and reasoning.

    Returns: {dimension_name: {"score": int, "reasoning": str}}
    """
    results = {}
    for dim in dimensions:
        # Find the dimension section
        pattern = rf'##\s*{re.escape(dim)}.*?Score:\s*(\d)'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            score = int(match.group(1))
            # Extract reasoning between dimension header and score
            reasoning_match = re.search(
                rf'##\s*{re.escape(dim)}.*?Reasoning:\s*(.*?)Score:',
                response, re.DOTALL | re.IGNORECASE
            )
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            results[dim] = {"score": min(max(score, 1), 5), "reasoning": reasoning}
        else:
            results[dim] = {"score": 3, "reasoning": "[PARSE FAILURE - defaulted to 3]"}

    return results
```

### Cross-Judge Aggregation

```python
def aggregate_pointwise(judge_results: list[dict],
                        dimensions: list[str],
                        holder_p: float = 0.4) -> dict:
    """Aggregate pointwise results across judges.

    Returns per-dimension averages, per-judge Holder means,
    overall mean, variance, and combined reasoning chains.
    """
    # Per-dimension: average across judges
    dim_averages = {}
    for dim in dimensions:
        scores = [jr[dim]["score"] for jr in judge_results if dim in jr]
        dim_averages[dim] = np.mean(scores) if scores else 0.0

    # Per-judge: Holder mean across dimensions
    judge_holder_scores = []
    for jr in judge_results:
        scores = [jr[dim]["score"] for dim in dimensions if dim in jr]
        judge_holder_scores.append(holder_mean(scores, p=holder_p))

    # Cross-judge: mean and variance
    overall_mean = np.mean(judge_holder_scores)
    overall_std = np.std(judge_holder_scores)

    # Reasoning chains: concatenate all
    all_reasoning = []
    for i, jr in enumerate(judge_results):
        chain = f"Judge {i+1}:\n"
        for dim in dimensions:
            if dim in jr:
                chain += f"  {dim}: {jr[dim]['reasoning']} [{jr[dim]['score']}/5]\n"
        all_reasoning.append(chain)

    return {
        "dimension_averages": dim_averages,
        "judge_holder_scores": judge_holder_scores,
        "mean": overall_mean,
        "std": overall_std,
        "reasoning_chains": "\n\n".join(all_reasoning),
    }
```

---

## 3. Pairwise Tournament

Used for final selection before stage handoff. Swiss-system brackets produce a
ranking in O(N log N) comparisons with O(log N) rounds.

### When to Use

- **During evolution:** Pointwise only (cheaper, provides dimensional diagnostic)
- **Final selection (before advancing to next stage):** Pairwise tournament on
  the top N candidates from the archive. Produces a more discriminating ranking
  at the quality boundary where pointwise scores are close.

### Swiss-System Algorithm

```python
def swiss_tournament(candidates: list, panel: JudgePanel,
                     num_rounds: int = None) -> list:
    """Run Swiss-system pairwise tournament.

    Args:
        candidates: List of content items to rank.
        panel: Judge panel for pairwise evaluation.
        num_rounds: Number of rounds (default: ceil(log2(N))).

    Returns:
        Candidates sorted by win record (best first).
    """
    n = len(candidates)
    if num_rounds is None:
        num_rounds = max(1, int(np.ceil(np.log2(n))))

    # Initialize records
    records = {i: {"wins": 0, "losses": 0, "opponents": set()}
               for i in range(n)}

    for round_num in range(num_rounds):
        # Pair candidates with similar records who haven't faced each other
        pairs = swiss_pairing(records)

        for i, j in pairs:
            # Run pairwise comparison with position bias mitigation
            winner = pairwise_compare(candidates[i], candidates[j], panel)
            records[winner]["wins"] += 1
            loser = j if winner == i else i
            records[loser]["losses"] += 1
            records[i]["opponents"].add(j)
            records[j]["opponents"].add(i)

    # Sort by wins (descending), then by strength-of-schedule
    ranked = sorted(records.keys(),
                    key=lambda x: (records[x]["wins"],
                                   sum(records[o]["wins"]
                                       for o in records[x]["opponents"])),
                    reverse=True)
    return [candidates[i] for i in ranked]
```

### Swiss Pairing

```python
def swiss_pairing(records: dict) -> list[tuple]:
    """Pair candidates with similar win records, avoiding rematches.

    Standard Swiss pairing: sort by wins, pair adjacent.
    If adjacent pair already played, swap with next available.
    """
    sorted_ids = sorted(records.keys(),
                        key=lambda x: records[x]["wins"],
                        reverse=True)
    pairs = []
    used = set()

    for i in range(0, len(sorted_ids) - 1, 2):
        a = sorted_ids[i]
        if a in used:
            continue

        # Find best available opponent (similar record, hasn't played a)
        for j in range(i + 1, len(sorted_ids)):
            b = sorted_ids[j]
            if b not in used and b not in records[a]["opponents"]:
                pairs.append((a, b))
                used.add(a)
                used.add(b)
                break

    # If odd number, last candidate gets a bye (counts as win)
    return pairs
```

### Pairwise Comparison with Position Bias Mitigation

```python
def pairwise_compare(story_a, story_b, panel: JudgePanel) -> int:
    """Compare two stories with position bias mitigation.

    Each judge evaluates twice: A-first and B-first.
    If a judge disagrees with itself, that comparison is ambiguous —
    run a third evaluation with explicit instruction to resolve.

    Returns: index of winner (0 = story_a, 1 = story_b).
    """
    votes_a = 0
    votes_b = 0

    for judge in panel.judges:
        # Evaluation 1: A first
        pref_1 = judge.compare(story_a, story_b)
        # Evaluation 2: B first
        pref_2 = judge.compare(story_b, story_a)

        if pref_1 == pref_2:
            # Consistent preference
            if pref_1 == "A":
                votes_a += 1
            else:
                votes_b += 1
        else:
            # Self-disagreement: run tiebreaker
            pref_3 = judge.compare_with_tiebreak(story_a, story_b)
            if pref_3 == "A":
                votes_a += 1
            else:
                votes_b += 1

    return 0 if votes_a > votes_b else 1
```

### Pairwise Judge Prompt

```
You are comparing two stories. Read both carefully, then decide which is
the stronger work overall.

STORY A:
---
{story_a_text}
---

STORY B:
---
{story_b_text}
---

First, briefly note the strengths and weaknesses of each story (2-3 sentences
per story). Then state your preference and explain why in 1-2 sentences.

Respond in this format:
STORY A ASSESSMENT: [strengths and weaknesses]
STORY B ASSESSMENT: [strengths and weaknesses]
PREFERENCE: [A or B]
REASON: [why]
```

The tiebreak prompt adds: "You previously evaluated these stories in different
orders and reached different conclusions. Please evaluate one final time, paying
careful attention to which story genuinely resonates more strongly."

---

## 4. Dynamic Rubrics

Story-specific evaluation criteria generated per-instance, overlaid on the
baseline 10 resonance dimensions. WritingBench found dynamic per-instance
criteria achieve 84% human alignment vs. 67% for static rubrics.

### Generation Prompt

```
Given the following story and its metadata, generate 3-5 evaluation criteria
specific to what THIS story is trying to achieve.

STORY:
{story_text}

METADATA (if available):
- Concept: {concept_premise}
- Target effect: {target_effect}
- Structure type: {structure_description}
- Voice spec: {voice_description}

Generate criteria that:
1. Are SPECIFIC to this story's ambitions (not generic quality measures)
2. Do NOT duplicate the baseline dimensions (Transportation, Suspense,
   Curiosity, Emotional Depth, Emotional Arc, Causal Coherence, Surprise,
   Ending Quality, Flow, Memorability)
3. Each has: a name, a one-sentence description, and score anchors for
   1/5, 3/5, and 5/5

Respond in this format:

CRITERION: [name]
DESCRIPTION: [one sentence]
1/5: [what this score looks like for this story]
3/5: [what this score looks like for this story]
5/5: [what this score looks like for this story]

CRITERION: [next]
...
```

### Redundancy Check

After generating dynamic criteria, check for overlap with the baseline 10:

```python
def check_redundancy(dynamic_criteria: list[dict],
                     baseline_dimensions: list[str]) -> list[dict]:
    """Remove dynamic criteria that overlap with baseline dimensions.

    Uses keyword matching as a fast heuristic. If a dynamic criterion's
    name or description contains >50% of a baseline dimension's keywords,
    it's likely redundant.
    """
    BASELINE_KEYWORDS = {
        "transportation": {"immersion", "absorption", "pull", "world"},
        "suspense": {"tension", "stakes", "uncertainty", "risk"},
        # ... etc for all 10
    }

    filtered = []
    for criterion in dynamic_criteria:
        text = (criterion["name"] + " " + criterion["description"]).lower()
        is_redundant = False
        for dim, keywords in BASELINE_KEYWORDS.items():
            overlap = sum(1 for k in keywords if k in text) / len(keywords)
            if overlap > 0.5:
                is_redundant = True
                break
        if not is_redundant:
            filtered.append(criterion)

    return filtered
```

### Integration

Dynamic criteria are scored alongside the baseline 10 dimensions. They use the
same 1-5 scale. Their scores are included in `public_metrics` and factor into
the Holder mean (giving them equal weight to baseline dimensions).

For stages where dynamic rubrics are disabled (e.g., Stage 1 concept evaluation),
only the baseline dimensions + stage-specific extras are used.

---

## 5. Quality Gates

Five checks run on each judge's output to detect degenerate evaluations. If a
judge fails a gate, their evaluation is discarded and optionally re-run.

### Gate 1: Score Clustering

**Check:** Standard deviation of scores across all dimensions within a single
judge's evaluation.

**Threshold:** SD < 0.3 = degenerate (judge gave essentially the same score
to everything).

```python
def check_score_clustering(scores: list[int]) -> bool:
    """Returns True if scores pass (sufficient variation)."""
    return np.std(scores) >= 0.3
```

### Gate 2: Reasoning Specificity

**Check:** Does the reasoning reference specific elements of the text, or is
it generic?

**Method:** Count proper nouns, quoted phrases, and specific references (line
numbers, paragraph references, character names) in the reasoning text.

**Threshold:** < 2 specific references per dimension = generic.

```python
def check_reasoning_specificity(reasoning: str, story_text: str) -> bool:
    """Returns True if reasoning is sufficiently specific."""
    # Count quoted phrases from the story
    quotes = re.findall(r'"([^"]{5,})"', reasoning)
    story_refs = sum(1 for q in quotes if q.lower() in story_text.lower())

    # Count character/place name references
    # (would use NER on story to extract names, then check reasoning)

    # Count specific structural references
    structural_refs = len(re.findall(
        r'\b(?:opening|ending|first|last|middle|paragraph|sentence|scene|'
        r'dialogue|line|passage|moment|section)\b',
        reasoning, re.IGNORECASE
    ))

    total_refs = story_refs + structural_refs
    return total_refs >= 2
```

### Gate 3: Slop in Reasoning

**Check:** The judge's own reasoning shouldn't contain Tier 1/2 slop words.
If the judge uses "tapestry" and "nuanced" in its evaluation, the persona
conditioning has failed.

**Threshold:** Any Tier 1 slop word in reasoning = flag.

```python
def check_reasoning_slop(reasoning: str) -> bool:
    """Returns True if reasoning is slop-free."""
    reasoning_lower = reasoning.lower()
    for word in TIER_1_WORDS:
        if re.search(rf'\b{word}\b', reasoning_lower):
            return False
    return True
```

### Gate 4: Cross-Story Repetition

**Check:** If the same judge produces very similar reasoning for different
stories, it's on autopilot.

**Method:** Embed reasoning chains for the last N stories evaluated by this
judge. Check pairwise cosine similarity.

**Threshold:** Mean pairwise similarity > 0.85 across last 5 stories = flag.

```python
def check_cross_story_repetition(current_reasoning: str,
                                  history: list[str],
                                  embed_fn) -> bool:
    """Returns True if reasoning is sufficiently unique."""
    if len(history) < 3:
        return True  # not enough history to check

    current_emb = embed_fn(current_reasoning)
    history_embs = [embed_fn(h) for h in history[-5:]]

    similarities = [cosine_similarity(current_emb, h) for h in history_embs]
    return np.mean(similarities) < 0.85
```

### Gate 5: Score-Reasoning Alignment

**Check:** Does the score match the reasoning? A judge that writes "this story
has a devastating ending" and then scores Ending Quality 2/5 is incoherent.

**Method:** Sentiment analysis of reasoning text vs. assigned score. Strongly
positive reasoning + low score (or vice versa) = misalignment.

**Threshold:** Sentiment-score correlation < 0.3 across dimensions = flag.

```python
def check_score_reasoning_alignment(results: dict,
                                     sentiment_fn) -> bool:
    """Returns True if scores align with reasoning sentiment."""
    scores = []
    sentiments = []
    for dim, data in results.items():
        scores.append(data["score"])
        sentiments.append(sentiment_fn(data["reasoning"]))

    if len(scores) < 3:
        return True

    correlation = np.corrcoef(scores, sentiments)[0, 1]
    return correlation > 0.3 or np.isnan(correlation)
```

### Gate Failure Handling

When a judge fails any gate:
1. Log the failure (gate type, judge persona, specific issue)
2. Discard the evaluation
3. Re-run with the same judge persona (up to 2 retries)
4. If 3 consecutive failures: flag the judge persona for review (possible
   prompt engineering issue or model degradation)

---

## 6. Output Format

### Pointwise Evaluation Result

```python
@dataclass
class PointwiseResult:
    """Result of pointwise evaluation for one content item."""
    dimension_scores: dict[str, float]    # per-dimension averages across judges
    judge_holder_scores: list[float]      # per-judge Holder mean
    combined_score: float                 # selection_score (mean + diversity bonus)
    judge_mean: float                     # mean of judge Holder scores
    judge_std: float                      # std of judge Holder scores
    reasoning_chains: str                 # concatenated judge reasoning
    quality_gate_passes: dict[str, bool]  # per-gate pass/fail
    dynamic_criteria: list[dict] | None   # if dynamic rubrics enabled
```

### Pairwise Tournament Result

```python
@dataclass
class TournamentResult:
    """Result of Swiss-system pairwise tournament."""
    rankings: list[int]                   # candidate indices, best first
    win_records: dict[int, dict]          # per-candidate {wins, losses, opponents}
    comparison_log: list[dict]            # detailed log of each comparison
    # Each comparison: {round, pair, votes, reasoning_excerpts}
```
