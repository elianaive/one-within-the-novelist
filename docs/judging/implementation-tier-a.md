# Judging Implementation: Tier A Anti-Slop Filter

The Tier A module is a deterministic NLP pipeline. No LLM calls. Input is raw
text; output is a composite slop score with per-filter breakdown. It runs on
every candidate at every stage (where enabled) as a cheap pre-filter before
expensive Tier B evaluation.

**Library:** spaCy with `en_core_web_sm` for tokenization, sentence boundary
detection, POS tagging, and lemmatization.

---

## 1. Pipeline Architecture

```
Input: raw text (string)
  │
  ├─ spaCy processing (one pass)
  │   ├─ Tokenization
  │   ├─ Sentence boundary detection
  │   ├─ POS tagging
  │   ├─ Lemmatization
  │   └─ Named entity recognition (for proper noun exclusion)
  │
  ├─ Filter 1: Banned Vocabulary          (uses: tokens, lemmas, NER)
  ├─ Filter 2: Construction Patterns      (uses: POS tags, dependency parse)
  ├─ Filter 3: Structural Patterns        (uses: sentences, paragraphs)
  ├─ Filter 4: Statistical Signals        (uses: tokens, sentences)
  ├─ Filter 5: Fiction Anti-Patterns      (uses: all of the above + dialogue extraction)
  ├─ Filter 6: Slop Trigrams              (uses: tokens)
  │
  └─ Composite Score (weighted combination of normalized filter scores)
```

**Key design decision:** Run spaCy once, share the parsed document across all
filters. The spaCy pass is the most expensive operation (~10-50ms per story);
filters are cheap operations on the parsed output.

---

## 2. Text Preprocessing

### spaCy Setup

```python
import spacy

nlp = spacy.load("en_core_web_sm")
# Disable components we don't need for speed
# Keep: tokenizer, tagger, parser, ner, lemmatizer
```

### Dialogue Extraction

Separate narrative text from dialogue for filters that treat them differently:

```python
import re

def extract_dialogue(text: str) -> tuple[list[str], str]:
    """Extract dialogue segments and return (dialogue_list, narrative_only).

    Handles: "quoted speech", 'single quotes', and em-dash dialogue.
    Returns dialogue segments and the text with dialogue removed.
    """
    dialogue_pattern = r'"[^"]*"|\'[^\']*\'|—[^—\n]*(?:—|\n)'
    dialogues = re.findall(dialogue_pattern, text)
    narrative = re.sub(dialogue_pattern, ' ', text)
    return dialogues, narrative
```

### Paragraph Segmentation

```python
def get_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs. Double newline = paragraph break."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return paragraphs
```

---

## 3. Filter 1: Banned Vocabulary

### Matching Rules

- **Case-insensitive** matching on lemmatized forms
- **Whole-word matching** only (not substrings: "deliver" does not match "delve")
- **Exclude proper nouns:** If a Tier 1 word appears as a named entity (person
  name, place name), it's not flagged. "Mr. Delve" is fine. "She delved into
  the mystery" is flagged.
- **Inflections included:** Lemmatization handles this — "delved," "delving,"
  "delves" all lemmatize to "delve"

### Tier 1 Words (Immediate Flag)

Overrepresented 50x+ in LLM output vs. human fiction baseline (Fiction-1B):

```python
TIER_1_WORDS = {
    "delve", "utilize", "leverage", "facilitate", "elucidate",
    "paradigm", "synergy", "multifaceted", "nuanced", "realm",
    "landscape", "myriad", "plethora", "embark", "endeavor",
    "tapestry", "camaraderie", "intricate", "testament", "holistic",
    "catalyze", "juxtapose", "illuminate", "underscore", "moreover",
    "furthermore", "nonetheless", "notably", "indeed", "comprehensive",
}
```

### Tier 2 Words (Suspicious in Clusters)

Overrepresented 10-50x. Flagged only when 3+ appear in the same 1,000-word
window:

```python
TIER_2_WORDS = {
    "robust", "innovative", "empower", "enhance", "foster",
    "pivotal", "crucial", "compelling", "resonate", "profound",
    "seamless", "vibrant", "dynamic", "meticulous", "unwavering",
    "poignant", "riveting", "captivating", "evocative", "visceral",
    "palpable", "tangible", "ephemeral", "ethereal", "luminescent",
    "gossamer", "ominous", "cacophony", "labyrinthine", "unfathomable",
}
```

### Tier 3 Phrases (Delete on Sight)

Full phrases that are near-exclusively LLM-generated:

```python
TIER_3_PHRASES = [
    "it's worth noting",
    "let's explore",
    "not just x, but y",  # detected by construction pattern filter
    "in today's world",
    "at the end of the day",
    "it is important to note",
    "this serves as a testament",
    "a testament to",
    "speaks volumes",
    "sends shivers down",
]
```

### Scoring

```python
def score_banned_vocabulary(doc, word_count: int) -> float:
    """Score 0-1 where higher = more slop detected.

    Returns density of banned vocabulary normalized to 0-1 range.
    """
    tier1_count = 0
    tier2_window_counts = []  # per 1000-word window
    tier3_count = 0

    # Count Tier 1 (lemmatized, excluding named entities)
    for token in doc:
        if token.ent_type_:  # skip named entities
            continue
        if token.lemma_.lower() in TIER_1_WORDS:
            tier1_count += 1

    tier1_density = tier1_count / max(word_count / 1000, 1)

    # Count Tier 2 in sliding windows
    # ... (sliding window of 1000 words, count Tier 2 per window)
    # Flag windows with 3+ Tier 2 words
    tier2_flagged_windows = sum(1 for c in tier2_window_counts if c >= 3)
    tier2_density = tier2_flagged_windows / max(len(tier2_window_counts), 1)

    # Count Tier 3 phrases (case-insensitive substring match)
    text_lower = doc.text.lower()
    for phrase in TIER_3_PHRASES:
        tier3_count += text_lower.count(phrase)
    tier3_density = tier3_count / max(word_count / 1000, 1)

    # Normalize to 0-1: weighted sum, capped at 1.0
    raw = (tier1_density * 0.5) + (tier2_density * 0.3) + (tier3_density * 0.2)
    return min(raw / 3.0, 1.0)  # 3.0 = calibration divisor (adjust empirically)
```

---

## 4. Filter 2: Construction Patterns

### Pattern Definitions

Each pattern is detected via POS tags, dependency parse, or regex on the
spaCy-processed document.

**"Not X, But Y" construction:**
```python
NOT_X_BUT_Y = re.compile(
    r'\bnot\s+(?:just\s+|merely\s+|only\s+)?[\w\s,]+,?\s*but\s+',
    re.IGNORECASE
)
# Threshold: >0 per 1,000 words in fiction = flag
```

**Body part + emotion shortcut:**
```python
BODY_PARTS = {"heart", "chest", "stomach", "gut", "spine", "jaw", "fist",
              "hand", "hands", "throat", "pulse", "blood"}
EMOTION_VERBS = {"hammer", "pound", "race", "clench", "tighten", "flutter",
                 "sink", "drop", "lurch", "twist", "knot", "squeeze",
                 "ache", "burn", "freeze", "tremble", "shiver"}
# Detect: BODY_PART + EMOTION_VERB in same clause
# Threshold: >2 per 1,000 words = flag
```

**Eyes as active subject:**
```python
EYES_AS_SUBJECT = re.compile(
    r'\b(?:his|her|their|the)\s+eyes?\s+(?:searched|scanned|traced|found|'
    r'locked|darted|swept|roamed|traveled|roved|bore into|bored into|'
    r'followed|drifted|flicked|narrowed on|settled on)',
    re.IGNORECASE
)
# Exclude involuntary: "eyes widened", "eyes filled" (these are fine)
# Threshold: >1 per 1,000 words = flag
```

**Hedging qualifiers:**
```python
HEDGING = re.compile(
    r'\b(?:seemed to|appeared to|as if|as though|kind of|sort of|'
    r'a hint of|a flicker of|a shadow of|somehow|something like)\b',
    re.IGNORECASE
)
# Threshold: >3 per 1,000 words = flag (human baseline ~1-2)
```

**Smile/voice as primary action:**
```python
SMILE_ACTION = re.compile(
    r'\ba?\s*(?:smile|grin|smirk|frown)\s+(?:spread|crept|played|tugged|'
    r'curved|crossed|touched|lit)',
    re.IGNORECASE
)
# Threshold: >1 per 1,000 words = flag
```

### Scoring

```python
def score_construction_patterns(text: str, word_count: int) -> float:
    """Score 0-1 where higher = more AI construction patterns detected."""
    scores = []

    not_x_but_y = len(NOT_X_BUT_Y.findall(text)) / max(word_count / 1000, 1)
    scores.append(min(not_x_but_y / 1.0, 1.0))  # any occurrence in fiction is suspicious

    # ... similar for each pattern ...

    return np.mean(scores)  # average across all pattern types
```

---

## 5. Filter 3: Structural Patterns

### Sentence Burstiness

```python
def score_burstiness(doc) -> float:
    """Measure sentence length variation. Low variation = AI-like.

    Human prose: CV 0.5-0.8+. AI prose: 0.2-0.4.
    Returns 0-1 where higher = more AI-like (less bursty).
    """
    sentence_lengths = [len(sent) for sent in doc.sents]
    if len(sentence_lengths) < 5:
        return 0.0  # too few sentences to measure

    mean_len = np.mean(sentence_lengths)
    std_len = np.std(sentence_lengths)

    if mean_len == 0:
        return 1.0

    cv = std_len / mean_len  # coefficient of variation

    # Map CV to 0-1 score: CV < 0.4 = high score (AI-like)
    if cv >= 0.6:
        return 0.0  # human-like variation
    elif cv <= 0.2:
        return 1.0  # very AI-like uniformity
    else:
        return 1.0 - (cv - 0.2) / 0.4  # linear interpolation
```

### Paragraph Length Uniformity

```python
def score_paragraph_uniformity(paragraphs: list[str]) -> float:
    """Measure paragraph length variation. Low variation = AI-like.

    Uses word count per paragraph. CV < 0.3 = flag.
    """
    if len(paragraphs) < 3:
        return 0.0

    lengths = [len(p.split()) for p in paragraphs]
    cv = np.std(lengths) / max(np.mean(lengths), 1)

    if cv >= 0.5:
        return 0.0
    elif cv <= 0.2:
        return 1.0
    else:
        return 1.0 - (cv - 0.2) / 0.3
```

### Consecutive Pronoun Starts

```python
SUBJECT_PRONOUNS = {"he", "she", "it", "they", "i", "we"}

def score_consecutive_pronouns(doc) -> float:
    """Flag 3+ consecutive sentences starting with the same pronoun."""
    sentences = list(doc.sents)
    max_consecutive = 0
    current_pronoun = None
    current_count = 0

    for sent in sentences:
        first_token = next((t for t in sent if not t.is_space), None)
        if first_token and first_token.text.lower() in SUBJECT_PRONOUNS:
            if first_token.text.lower() == current_pronoun:
                current_count += 1
                max_consecutive = max(max_consecutive, current_count)
            else:
                current_pronoun = first_token.text.lower()
                current_count = 1
        else:
            current_pronoun = None
            current_count = 0

    # 3+ consecutive = flag
    if max_consecutive >= 5:
        return 1.0
    elif max_consecutive >= 3:
        return 0.6
    else:
        return 0.0
```

### Em-Dash Overload

```python
def score_em_dash(text: str, word_count: int) -> float:
    """Flag em-dash overuse. >2 per 250 words (approx. 1 page) = flag."""
    em_dashes = text.count('—') + text.count('--')
    pages = max(word_count / 250, 1)
    per_page = em_dashes / pages

    if per_page <= 2:
        return 0.0
    elif per_page >= 5:
        return 1.0
    else:
        return (per_page - 2) / 3
```

---

## 6. Filter 4: Statistical Signals

### MATTR-500 (Lexical Diversity)

Moving Average Type-Token Ratio with window size 500. More stable than
simple TTR across different text lengths.

```python
def mattr(tokens: list[str], window_size: int = 500) -> float:
    """Compute Moving Average Type-Token Ratio.

    Returns ratio 0-1. Literary fiction: >0.80. AI: 0.55-0.75.
    """
    if len(tokens) < window_size:
        # Fall back to simple TTR for short texts
        return len(set(tokens)) / max(len(tokens), 1)

    ttrs = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        ttrs.append(len(set(window)) / window_size)

    return np.mean(ttrs)

def score_mattr(doc) -> float:
    """Score 0-1 where higher = more AI-like (less diverse)."""
    tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]
    m = mattr(tokens)

    if m >= 0.80:
        return 0.0  # literary-quality diversity
    elif m <= 0.60:
        return 1.0  # very AI-like repetition
    else:
        return 1.0 - (m - 0.60) / 0.20
```

### Personal Pronoun Density

```python
PERSONAL_PRONOUNS = {"i", "me", "my", "mine", "myself",
                     "you", "your", "yours", "yourself",
                     "he", "him", "his", "himself",
                     "she", "her", "hers", "herself",
                     "we", "us", "our", "ours", "ourselves",
                     "they", "them", "their", "theirs", "themselves"}

def score_pronoun_density(doc) -> float:
    """Score 0-1 where higher = fewer pronouns (AI-like).

    AI uses fewer personal pronouns, which correlates with reduced
    reader transportation (Voss et al., Nature 2025).
    Human fiction baseline: ~60-80 per 1,000 words.
    AI fiction: ~40-55 per 1,000 words.
    """
    word_count = len([t for t in doc if not t.is_punct and not t.is_space])
    pronoun_count = sum(1 for t in doc if t.text.lower() in PERSONAL_PRONOUNS)
    density = pronoun_count / max(word_count / 1000, 1)

    if density >= 65:
        return 0.0  # human-like pronoun usage
    elif density <= 40:
        return 1.0  # very AI-like pronoun scarcity
    else:
        return 1.0 - (density - 40) / 25
```

### Nominalization Rate

```python
NOMINALIZATION_SUFFIXES = ("tion", "ment", "ness", "ity", "ence", "ance")

def score_nominalization(doc) -> float:
    """Score 0-1. AI prose over-nominalizes (1.5-2x human rate, d=1.23).

    Detects -tion, -ment, -ness, -ity, -ence, -ance suffixes on
    words that are NOT themselves common nouns (exclude: "station",
    "moment", "business", etc. via POS check — flag only when the
    word is a nominalization of a verb/adjective).
    """
    word_count = len([t for t in doc if not t.is_punct and not t.is_space])
    nom_count = 0

    for token in doc:
        if token.pos_ == "NOUN" and len(token.text) > 5:
            if token.text.lower().endswith(NOMINALIZATION_SUFFIXES):
                # Check if it's a true nominalization (has a verb/adj lemma root)
                # Simple heuristic: exclude very common nouns
                nom_count += 1

    density = nom_count / max(word_count / 1000, 1)

    # Human baseline: ~15-25 per 1,000 words
    # AI baseline: ~30-50 per 1,000 words
    if density <= 25:
        return 0.0
    elif density >= 45:
        return 1.0
    else:
        return (density - 25) / 20
```

---

## 7. Filter 5: Fiction Anti-Patterns (Nous Research)

These are higher-level patterns that require combining multiple signals. They
run only on candidates that pass filters 1-4 (to avoid wasting computation on
already-flagged text).

### Over-Explanation (The #1 Problem)

```python
def detect_over_explanation(doc, dialogues: list[str]) -> float:
    """Detect show-then-tell pattern.

    Pattern: Scene shows emotion through action/dialogue →
    narrator immediately restates the emotion explicitly.
    """
    EMOTION_ADJECTIVES = {
        "angry", "sad", "happy", "afraid", "scared", "nervous",
        "frustrated", "relieved", "anxious", "terrified", "furious",
        "devastated", "elated", "confused", "desperate", "hopeless",
        "excited", "ashamed", "guilty", "jealous", "lonely",
    }
    TELL_VERBS = {"felt", "was", "seemed", "appeared", "looked", "realized"}

    sentences = list(doc.sents)
    over_explain_count = 0

    for i in range(len(sentences) - 1):
        sent = sentences[i]
        next_sent = sentences[i + 1]

        # Check if current sentence contains action/dialogue
        has_action = any(t.pos_ == "VERB" and t.dep_ == "ROOT" for t in sent)

        # Check if next sentence is an emotion tell
        next_tokens = [t.text.lower() for t in next_sent]
        has_tell_verb = any(t in TELL_VERBS for t in next_tokens)
        has_emotion = any(t in EMOTION_ADJECTIVES for t in next_tokens)

        if has_action and has_tell_verb and has_emotion:
            over_explain_count += 1

    word_count = len([t for t in doc if not t.is_punct and not t.is_space])
    density = over_explain_count / max(word_count / 1000, 1)

    return min(density / 2.0, 1.0)  # 2+ per 1k words = max flag
```

### Dialogue Naturalness

```python
def score_dialogue_naturalness(dialogues: list[str]) -> float:
    """Detect AI dialogue patterns: too grammatical, too long, no interruptions.

    Human dialogue: fragments, interruptions, false starts, <15 words avg.
    AI dialogue: complete sentences, perfect grammar, >15 words avg.
    """
    if not dialogues:
        return 0.0

    lengths = [len(d.split()) for d in dialogues]
    mean_length = np.mean(lengths)

    # Check for interruptions/fragments
    fragments = sum(1 for d in dialogues if not d.rstrip('."\'!?').endswith(('.', '!', '?')))
    fragment_ratio = fragments / max(len(dialogues), 1)

    score = 0.0
    if mean_length > 20:
        score += 0.5  # too long
    elif mean_length > 15:
        score += 0.25

    if fragment_ratio < 0.1:
        score += 0.3  # no fragments = too perfect
    if fragment_ratio < 0.05:
        score += 0.2

    return min(score, 1.0)
```

### Scene-Summary Imbalance

```python
def score_scene_summary_ratio(doc) -> float:
    """Flag if <70% of prose is in-scene (vs. summary/exposition).

    Heuristic: in-scene = contains dialogue OR concrete action verbs
    with specific subjects. Summary = abstract statements, time
    compression, exposition.
    """
    paragraphs = get_paragraphs(doc.text)
    if not paragraphs:
        return 0.0

    scene_paras = 0
    for para in paragraphs:
        has_dialogue = '"' in para or "'" in para
        has_action = bool(re.search(r'\b(?:said|asked|walked|ran|opened|closed|picked|grabbed|turned|looked)\b', para, re.IGNORECASE))
        if has_dialogue or has_action:
            scene_paras += 1

    ratio = scene_paras / len(paragraphs)

    if ratio >= 0.7:
        return 0.0
    elif ratio <= 0.4:
        return 1.0
    else:
        return 1.0 - (ratio - 0.4) / 0.3
```

---

## 8. Filter 6: Slop Trigrams

The 433 three-word phrases statistically overrepresented in LLM output
(EQ-Bench). A representative subset of the most diagnostic trigrams:

```python
SLOP_TRIGRAMS = {
    # Sensation
    "voice barely whisper", "took deep breath", "heart pounding chest",
    "breath caught throat", "blood ran cold", "chill run spine",
    "knuckles turning white", "heart skipped beat", "tears streaming face",
    "hands trembling slightly", "stomach dropped floor",
    # Emotional
    "growing sense unease", "mind racing possibilities", "wave relief washed",
    "pang guilt shot", "surge anger rose", "mix emotions swirled",
    "weight world shoulders", "lump formed throat",
    # Scene-setting
    "dimly lit room", "air thick tension", "casting long shadows",
    "sun dipped horizon", "dust motes danced", "silence hung heavy",
    "moonlight streaming window", "cold wind howled",
    # Plot formula
    "one thing certain", "whatever lay ahead", "would never same",
    "couldn't shake feeling", "something about way", "moment changed everything",
    "truth finally dawned", "pieces fell place",
    # Voice/body
    "smile playing lips", "eyes locked across", "door creaked open",
    "room fell silent", "footsteps echoed hallway", "voice barely audible",
}
# Full list: 433 trigrams. The above are the top ~60 most diagnostic.
# Complete list to be compiled from EQ-Bench slop-score repository.

def score_slop_trigrams(text: str, word_count: int) -> float:
    """Count slop trigram occurrences, normalize to 0-1."""
    text_lower = text.lower()
    count = sum(1 for trigram in SLOP_TRIGRAMS if trigram in text_lower)
    density = count / max(word_count / 1000, 1)

    if density <= 1:
        return 0.0
    elif density >= 5:
        return 1.0
    else:
        return (density - 1) / 4
```

---

## 9. Composite Score

### Normalization

Each filter returns a score in [0, 1] where 0 = human-like and 1 = maximally
AI-like. No further normalization needed — all filters output on the same scale.

### Weighted Combination

```python
COMPOSITE_WEIGHTS = {
    "banned_vocabulary": 0.20,
    "construction_patterns": 0.20,
    "burstiness": 0.15,
    "mattr": 0.15,
    "pronoun_density": 0.10,
    "structural_patterns": 0.10,  # paragraph uniformity + pronoun starts + em-dash
    "slop_trigrams": 0.05,
    "fiction_anti_patterns": 0.05,  # over-explain + dialogue + scene-summary
}

def composite_slop_score(filter_scores: dict[str, float]) -> float:
    """Weighted combination of all filter scores. Returns 0-1."""
    total = sum(
        filter_scores.get(name, 0.0) * weight
        for name, weight in COMPOSITE_WEIGHTS.items()
    )
    return total
```

### Rejection Threshold

**Default: 0.35**

- Score < 0.2: Very clean — likely human-like or well-constrained LLM output
- Score 0.2-0.35: Some AI signals but within acceptable range
- Score > 0.35: Too many AI signals — reject before Tier B evaluation
- Score > 0.5: Heavy slop — immediate rejection

The threshold is configurable per stage (via `tier_a_threshold` in stage config).
Stages with tighter quality requirements can lower the threshold.

---

## 10. Calibration Against Baselines

### Procedure

1. Run Tier A on 500 passages from Fiction-1B (human fiction baseline)
2. Run Tier A on 500 passages from the same prompts generated by current LLMs
3. Compute the composite score distribution for each set
4. Adjust filter thresholds and composite weights so that:
   - 95%+ of human passages score below the rejection threshold
   - 50%+ of LLM passages score above the rejection threshold
5. The 5% false positive rate on human text is acceptable — some human writing
   genuinely resembles LLM patterns (particularly genre fiction)

### Per-Filter Calibration

For each individual filter, verify against baselines:
- The filter's score distribution for human text should be clearly separated
  from LLM text
- If a filter doesn't discriminate (similar distributions for human and LLM),
  reduce its weight in the composite or remove it
- If a filter over-triggers on human text, adjust its internal thresholds

### Escalation

As part of the coevolving judges mechanism (see implementation-scoring.md):
when the system's median combined_score exceeds 3.5, tighten Tier A thresholds
by 10%. This raises the floor as the population improves.

---

## 11. Output Format

```python
@dataclass
class TierAResult:
    composite_score: float          # 0-1, weighted combination
    passed: bool                    # composite_score < threshold
    filter_scores: dict[str, float] # per-filter breakdown
    flagged_items: list[dict]       # specific flagged instances with context

    # Example flagged_item:
    # {"filter": "banned_vocabulary", "word": "tapestry", "tier": 1,
    #  "location": "paragraph 3, sentence 2", "context": "...the tapestry of..."}
```

The `flagged_items` list provides human-readable explanations of what was
detected and where. This is useful for debugging and for the episodic memory
system (understanding why a story was rejected).
