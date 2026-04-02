"""Filter 1: Banned Vocabulary.

Words and phrases statistically overrepresented in LLM output relative to
human writing. Sources: Nous Research ANTI-SLOP.md, EQ-Bench slop-forensics,
Reinhart et al. (PNAS 2025), Paech et al. (ICLR 2026), Wikipedia "Signs of
AI Writing".
"""

# ---------------------------------------------------------------------------
# Tier 1: 50x+ overrepresented vs human fiction (Reinhart et al. Table 1).
# Any occurrence = flag. High density = reject.
# ---------------------------------------------------------------------------

TIER_1_WORDS = {
    # From spec (Nous ANTI-SLOP.md)
    "delve", "utilize", "leverage", "facilitate", "elucidate",
    "paradigm", "synergy", "synergize", "multifaceted", "nuanced",
    "realm", "landscape", "myriad", "plethora", "embark", "endeavor",
    "encompass", "tapestry", "camaraderie", "intricate", "testament",
    "holistic", "catalyze", "catalyst", "juxtapose",
    # Reinhart et al. Table 1: 50x+ overrepresentation (GPT-4o / GPT-4o Mini)
    "unspoken",     # 102x GPT-4o
    "amidst",       # 100x GPT-4o, 90x 4o-Mini
    "palpable",     # 95x GPT-4o, 145x 4o-Mini
    "solace",       # 95x GPT-4o
    "fleeting",     # 84x GPT-4o, 124x 4o-Mini
    "unravel",      # 83x GPT-4o
    "grapple",      # 131x 4o-Mini
    "ignite",       # 122x 4o-Mini
    "cacophony",    # 89x 4o-Mini
    "vibrant",      # 92x 4o-Mini
}

# ---------------------------------------------------------------------------
# Tier 2: Suspicious in clusters. One is fine; 3+ in a 1000-word window = flag.
# 10-50x overrepresented. Source: spec, Reinhart et al.
# ---------------------------------------------------------------------------

TIER_2_WORDS = {
    # From spec
    "robust", "comprehensive", "seamless", "cutting-edge", "innovative",
    "streamline", "empower", "foster", "enhance", "elevate", "optimize",
    "scalable", "pivotal", "profound", "resonate", "underscore", "harness",
    "navigate", "cultivate", "bolster", "galvanize", "cornerstone",
    "game-changer",
    # Reinhart et al. — Llama instruction-tuned models 20-50x
    "unease",       # 63-101x across Llama instruct models
    "pang",         # 25-29x Llama instruct
    "prioritize",   # 24-27x Llama instruct
    "waft",         # 24x Llama 8B instruct
}

# ---------------------------------------------------------------------------
# Fiction-specific vocabulary flags
# ---------------------------------------------------------------------------

# Sensation/atmosphere overuse. Cluster detection (Tier 2 rules).
# Source: spec §1 "Fiction-Specific Vocabulary Flags"
FICTION_ATMOSPHERE_WORDS = {
    "ethereal", "luminescent", "ominous", "crystalline", "gossamer",
    "iridescent", "visceral", "ephemeral", "resplendent",
    # Paech et al. Appendix G — descriptors overused across 60%+ of 67 models
    "eerie", "enigmatic", "eldritch", "tenebrous", "otherworldly",
}

# Flattery/tourism language. Any occurrence = flag in fiction.
FICTION_TOURISM_WORDS = {
    "fascinating", "majestic", "captivating", "breathtaking", "stunning",
}

FICTION_TOURISM_PHRASES = [
    "rich cultural heritage", "stands as a testament", "lasting impact",
    "vital role", "watershed moment",
]

# Weasel attribution phrases (should never appear in fiction).
WEASEL_PHRASES = [
    "industry reports suggest", "observers note", "experts agree",
    "studies show", "research suggests", "according to experts",
]

# ---------------------------------------------------------------------------
# Tier 3: Filler phrases — verbal tics LLMs insert reflexively.
# Sources: spec, Nous ANTI-SLOP.md, Nous evaluate.py regex patterns.
# ---------------------------------------------------------------------------

TIER_3_PHRASES = [
    "it's worth noting",
    "it is important to note",
    "importantly,",
    "notably,",
    "interestingly,",
    "let's dive into",
    "let's explore",
    "furthermore,",
    "moreover,",
    "additionally,",
    "in today's",
    "at the end of the day",
    "it goes without saying",
    "when it comes to",
    "in the realm of",
    "one might argue",
    "it could be suggested",
    "this begs the question",
    "a comprehensive approach",
    "a holistic approach",
    "a nuanced approach",
    "this serves as a testament",
    "a testament to",
    "speaks volumes",
    "sends shivers down",
    "in conclusion,",
    "to summarize,",
    # Additional from Nous ANTI-SLOP.md (not in spec)
    "as we can see",
    "in this section",
    "as mentioned earlier",
    "without further ado",
]

# ---------------------------------------------------------------------------
# Fiction AI Tells — regex patterns for fiction-specific LLM clichés.
# Source: Nous autonovel evaluate.py FICTION_AI_TELLS.
# These are scored separately from vocabulary (detected by regex, not lemma).
# ---------------------------------------------------------------------------

FICTION_AI_TELL_PATTERNS = [
    r"a sense of \w+",
    r"couldn'?t help but feel",
    r"the weight of \w+",
    r"the air was thick with",
    r"eyes widened",
    r"a wave of \w+ washed over",
    r"a pang of \w+",
    r"heart pounded in (?:his|her|their) chest",
    r"(?:raven|dark|golden|silver) (?:hair|tresses) (?:spilled|cascaded|tumbled|fell)",
    r"piercing (?:blue|green|gray|grey|dark) eyes",
    r"a knowing (?:smile|grin|look|glance)",
    r"(?:he|she|they) felt a (?:surge|rush|wave|pang|flicker) of",
    r"the silence (?:was|hung|stretched|grew) (?:heavy|thick|oppressive|deafening)",
    r"let out a breath (?:he|she|they) didn'?t (?:know|realize)",
    r"something (?:dark|ancient|primal|unnamed) stirred",
]

# ---------------------------------------------------------------------------
# Telling adverbs — emotion-adverbs that signal "telling" not "showing".
# Source: Nous autonovel evaluate.py TELLING_PATTERNS.
# High density of these = narrator naming emotions instead of showing them.
# ---------------------------------------------------------------------------

TELLING_ADVERBS = {
    "angrily", "sadly", "happily", "nervously", "excitedly",
    "desperately", "furiously", "anxiously", "guiltily", "bitterly",
    "wearily", "miserably", "longingly", "jealously", "resentfully",
    "fearfully", "gleefully", "sorrowfully", "triumphantly", "dejectedly",
}

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

import re

_COMPILED_FICTION_TELLS = [re.compile(p, re.IGNORECASE) for p in FICTION_AI_TELL_PATTERNS]


def score_banned_vocabulary(doc, word_count: int) -> tuple[float, list[dict]]:
    """Score 0-1 where higher = more slop. Returns (score, flagged_items).

    Combines: Tier 1 words, Tier 2 cluster detection, Tier 3 phrases,
    fiction tourism, weasel phrases, fiction AI tells, and telling adverbs.
    """
    flagged = []
    tier1_count = 0
    tier2_counts: list[int] = []
    tier3_count = 0
    fiction_tell_count = 0
    telling_adverb_count = 0

    all_tier1 = TIER_1_WORDS | FICTION_TOURISM_WORDS
    all_tier2 = TIER_2_WORDS | FICTION_ATMOSPHERE_WORDS

    # Tier 1 + fiction tourism: lemmatized, excluding named entities
    for token in doc:
        if token.ent_type_:
            continue
        lemma = token.lemma_.lower()
        if lemma in all_tier1:
            tier1_count += 1
            flagged.append({
                "filter": "banned_vocabulary", "tier": 1,
                "word": token.text, "lemma": lemma,
            })
        # Telling adverbs
        if token.text.lower() in TELLING_ADVERBS:
            telling_adverb_count += 1

    tier1_density = tier1_count / max(word_count / 1000, 1)

    # Tier 2 + fiction atmosphere: sliding 1000-word windows
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    window_size = 1000
    if len(tokens) >= window_size:
        for start in range(0, len(tokens) - window_size + 1, window_size // 2):
            window = tokens[start:start + window_size]
            count = sum(
                1 for t in window
                if not t.ent_type_ and t.lemma_.lower() in all_tier2
            )
            tier2_counts.append(count)
    else:
        count = sum(
            1 for t in tokens
            if not t.ent_type_ and t.lemma_.lower() in all_tier2
        )
        tier2_counts.append(count)

    tier2_flagged_windows = sum(1 for c in tier2_counts if c >= 3)
    tier2_density = tier2_flagged_windows / max(len(tier2_counts), 1)

    # Tier 3 + weasel + tourism phrases
    text_lower = doc.text.lower()
    all_phrases = TIER_3_PHRASES + FICTION_TOURISM_PHRASES + WEASEL_PHRASES
    for phrase in all_phrases:
        n = text_lower.count(phrase)
        if n > 0:
            tier3_count += n
            flagged.append({
                "filter": "banned_vocabulary", "tier": 3,
                "phrase": phrase, "count": n,
            })

    tier3_density = tier3_count / max(word_count / 1000, 1)

    # Fiction AI tells (regex patterns)
    for pattern in _COMPILED_FICTION_TELLS:
        matches = pattern.findall(doc.text)
        if matches:
            fiction_tell_count += len(matches)
            flagged.append({
                "filter": "banned_vocabulary", "tier": "fiction_tell",
                "pattern": pattern.pattern, "count": len(matches),
            })

    fiction_tell_density = fiction_tell_count / max(word_count / 1000, 1)

    # Telling adverbs (human baseline ~1-2 per 1k; AI 4-8 per 1k)
    telling_density = telling_adverb_count / max(word_count / 1000, 1)

    # Composite: weighted sum of all components, capped at 1.0
    # Tiers 1-3: same weights as before (0.5 + 0.3 + 0.2 = 1.0, /3.0)
    # Fiction tells and telling adverbs folded into the overall score
    base_raw = (tier1_density * 0.5) + (tier2_density * 0.3) + (tier3_density * 0.2)
    base_score = min(base_raw / 3.0, 1.0)

    fiction_score = min(fiction_tell_density / 4.0, 1.0)
    telling_score = min(max(telling_density - 2, 0) / 4.0, 1.0)

    # Blend: 70% vocabulary tiers, 20% fiction tells, 10% telling adverbs
    combined = (base_score * 0.70) + (fiction_score * 0.20) + (telling_score * 0.10)
    return min(combined, 1.0), flagged
