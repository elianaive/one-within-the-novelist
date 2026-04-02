"""Filter 6: Slop N-grams (Trigrams & Bigrams).

Three-word and two-word phrases statistically overrepresented in LLM output.
Source: EQ-Bench slop-score (sam-paech/slop-score), 430 trigrams + 200 bigrams
compared against human text baseline.

The lists split into two categories:
- Fiction slop: overused narrative phrases ("voice barely whisper", "heart pounding chest")
- Essay/business slop: should never appear in fiction ("plays crucial role", "long term success")
"""

# ---------------------------------------------------------------------------
# 430 slop trigrams — complete list from EQ-Bench slop-score
# ---------------------------------------------------------------------------

SLOP_TRIGRAMS = {
    # Fiction: sensation/action
    "voice barely whisper", "took deep breath", "heart pounding chest",
    "breath caught throat", "blood ran cold", "chill run spine",
    "knuckles turning white", "heart skipped beat", "tears streaming face",
    "hands trembling slightly", "stomach dropped floor", "eyes wide fear",
    "casting long shadows", "sun dipped horizon", "smile playing lips",
    "dust motes danced", "room fell silent", "door creaked open",
    "said voice barely", "help feel sense", "said voice low",
    "voice barely audible", "asked voice barely", "could shake feeling",
    "could help feel", "said voice steady", "air thick scent",
    "long shadows across", "days turned weeks", "felt chill run",
    "whispered voice barely", "felt like eternity", "shiver run spine",
    "said voice filled", "spreading across face", "leaned back chair",
    "voice steady despite", "unlike anything ever", "felt shiver run",
    "taking deep breath", "ready face whatever", "heart pounded chest",
    "trying make sense", "dipped horizon casting", "said voice trembling",
    "something else something", "deep breath trying", "asked voice trembling",
    "could feel weight", "said voice firm", "felt strange sense",
    "words hung air", "brow furrowed concentration", "challenges lay ahead",
    "something else entirely", "voice low rumble", "sun began set",
    "sent shiver spine", "end end end", "voice trembling slightly",
    "eyes never leaving", "growing sense unease", "door creaked open",
    "shake feeling something", "air hung thick", "heart skipped beat",
    "made way back", "took step back", "rain continued fall",
    "sun hung low", "said voice dripping", "said voice soft",
    "casting warm glow", "took step forward", "eyes wide fear",
    "renewed sense purpose", "would find way", "trying keep voice",
    "hung low sky", "horizon casting long", "smile spreading across",
    "leaned forward eyes", "anything ever seen", "one thing certain",
    "mind racing possibilities", "low sky casting", "said voice laced",
    "heart hammered ribs", "hung heavy air", "looked eyes filled",
    "face whatever challenges", "took step closer", "eyes filled mixture",
    "road ahead would", "spread like wildfire", "whispered voice trembling",
    "felt sense peace", "hung air heavy", "air thick tension",
    "newfound sense purpose", "flicker something akin", "grin spreading across",
    "deep breath stepped", "door swung open", "knew would never",
    "eyes locked onto", "tried make sense", "dust motes danced",
    "dimly lit room", "small intricately carved", "brow furrowed confusion",
    "deep breath steeling", "could help notice", "one last time",
    "eyes darting around", "sent shivers spine", "keep voice steady",
    "young woman named", "said voice tinged", "painting sky hues",
    "eyes wide wonder", "nodded mind racing", "fingers flying across",
    "whatever challenges lay", "figure emerged shadows", "mind raced trying",
    "knew road ahead", "raised hand silencing", "piercing blue eyes",
    "felt strange sensation", "turned walked away", "said voice calm",
    "small smile playing", "felt cold dread", "mind racing questions",
    "could find way", "spread across face", "first time long",
    "ground beneath feet", "hung thick scent", "sky casting long",
    "mind racing implications", "weeks weeks months", "asked voice low",
    "chill ran spine", "said trying sound", "would never forget",
    "gaze sweeping across", "asked voice laced", "shiver ran spine",
    "turned weeks weeks", "spent countless hours", "brow furrowed concern",
    "eyes wide mixture", "resonated deep within", "eyes widened surprise",
    "air thick anticipation", "breath catching throat", "mind racing tried",
    "deep breath feeling", "looked around room", "shook head trying",
    "whatever lay ahead", "said voice like", "replied voice steady",
    "eyes scanning room", "felt growing sense", "stepped forward voice",
    "never seen anything", "stammered voice barely", "help feel twinge",
    "knew one thing", "stepped forward eyes", "smile spread across",
    "maybe maybe could", "feel sense unease", "seen anything like",
    "knew chilling certainty", "growing sense dread", "intricately carved wooden",
    "heart heavy weight", "words echoed mind", "sighed running hand",
    "scent damp earth", "air thick smell", "said voice smooth",
    "began set casting", "blood ran cold", "said voice echoing",
    "hand instinctively reaching", "sense peace wash", "elara said voice",
    "smile tugging corners", "said voice tight", "could feel power",
    "eyes wide terror", "mind already racing", "knuckles turning white",
    "felt glimmer hope", "made decision would", "faint almost imperceptible",
    "eyes widened shock", "feel sense pride", "tears streaming face",
    "horizon painting sky", "casting eerie glow", "long could remember",
    "continued fall washing",
    # Essay/business slop — should never appear in fiction
    "decision making processes", "long term success", "plays crucial role",
    "make informed decisions", "plays pivotal role", "supply chain management",
    "corporate social responsibility", "play crucial role",
    "long term sustainability", "play pivotal role", "one size fits",
    "ethical decision making", "work life balance", "key performance indicators",
    "extends far beyond", "long term viability", "real time data",
    "decision making process", "loyal customer base", "driven decision making",
    "played crucial role", "play significant role", "plays vital role",
    "data driven decision", "culture continuous improvement",
    "user generated content", "played pivotal role", "play vital role",
    "address root causes", "one must first", "double edged sword",
    "provide valuable insights", "pivotal role shaping",
    "informed decision making", "strategic decision making",
    "data driven approach", "plays critical role", "plays significant role",
    "extend far beyond", "compelling case study", "crucial role shaping",
    "multi faceted approach", "played significant role",
    "identify areas improvement", "cross functional teams",
    "maintain competitive edge", "short term gains",
    "long term consequences", "another critical aspect",
    "increasingly interconnected world", "extends beyond mere",
    "addressing root causes", "essay explores multifaceted",
    "changing market conditions", "culture continuous learning",
    "environmental social governance", "long term growth",
    "play critical role", "changing consumer preferences",
    "real time monitoring", "public awareness campaigns",
    "public private partnerships", "requires multifaceted approach",
    "multi pronged approach", "significant role shaping",
    "must move beyond", "long term value", "far reaching consequences",
    "size fits approach", "various factors including",
    "provides valuable insights", "rapid technological advancements",
    "landscape continues evolve", "fostering culture continuous",
    "cross functional collaboration", "shifting consumer preferences",
    "long term benefits", "world word count", "ensuring long term",
    "critical thinking skills", "problem solving skills",
    "merely academic exercise", "small medium sized",
    "randomized controlled trials", "evolving consumer preferences",
    "manifest various ways", "long term strategic", "real time feedback",
    "essay delve multifaceted", "one primary reasons",
    "rapid technological change", "positive work environment",
    "essay aims explore", "creating virtuous cycle", "future word count",
    "testament enduring power", "success word count", "long term vision",
    "long term economic", "provide real time", "landscape modern business",
    "navigate complexities modern", "requires careful consideration",
    "among team members", "make informed choices", "new revenue streams",
    "rapidly evolving landscape", "medium sized enterprises",
    "collaborative problem solving", "general data protection",
    "ever evolving landscape", "sustainable competitive advantage",
    "significant turning point", "faces several challenges",
    "made significant strides", "mitigate risks associated",
    "far reaching implications", "ethical business practices",
    "across various sectors", "evidence based strategies",
    "cutting edge technology", "adapt changing market",
    "addressing challenges requires", "marked turning point",
    "essay explore multifaceted", "industry continues evolve",
    "marked significant turning", "complex interplay factors",
    "extend beyond individual", "marked pivotal moment",
    "built upon foundation", "offers valuable insights",
    "extends beyond individual", "closer examination reveals",
    "move beyond simply", "foster sense community",
    "contemporary business landscape", "manifest various forms",
    "serves stark reminder", "driving force behind",
    "today's fast paced", "cost benefit analysis",
    "faced significant challenges", "fostering sense community",
    "gain deeper understanding", "another critical component",
    "another critical factor", "requires deep understanding",
    "essay explores significance", "maintaining competitive edge",
    "characterized rapid technological", "rise social media",
    "far beyond simple", "also plays crucial", "long term commitment",
    "enhance operational efficiency", "tapestry woven threads",
    "ability adapt changing", "complex tapestry woven",
    "evidence based approach", "navigate complex landscape",
    "fostering sense belonging", "era defined rapid",
    "forward thinking approach", "critical role shaping",
    "fostering long term", "based decision making",
    "unlock full potential", "moving beyond simple",
    "several key areas", "without fear reprisal",
    "long term outcomes", "rapidly changing world",
    "high stakes environment", "goes beyond simply",
    "professional development opportunities", "creates virtuous cycle",
    "another layer complexity", "growing body research",
    "requires multi pronged", "better health outcomes",
    "extend beyond immediate", "evidence based decision",
    "one primary challenges", "essential long term",
    "level playing field", "strong brand identity",
    "delicate balancing act", "crucial step towards",
    "profound far reaching", "unique value proposition",
    "distinct yet interconnected", "long term stability",
    "threat new entrants", "serves cautionary tale",
    "thinking problem solving", "political economic social",
    "long term goals", "beyond surface level",
    "delivering high quality", "data driven decisions",
    "make data driven", "stakeholders including employees",
    "customer satisfaction loyalty", "move beyond simplistic",
    "fostering sense shared", "foster long term",
    "however crucial acknowledge", "customer centric approach",
    "modern business landscape", "far beyond mere",
    "technological advancements evolving", "meticulous attention detail",
    "across various industries", "played vital role",
    "deeply embedded within", "creates fertile ground",
    "long term impact", "fosters sense shared",
    "long term investment", "confront uncomfortable truths",
    "presents significant challenge", "complex often contradictory",
    "ensure long term", "extending far beyond",
    "goes far beyond", "extended far beyond",
    "success inextricably linked", "company invested heavily",
    "challenge status quo",
}

# ---------------------------------------------------------------------------
# 200 slop bigrams — complete list from EQ-Bench slop-score
# ---------------------------------------------------------------------------

SLOP_BIGRAMS = {
    "said voice", "deep breath", "voice barely", "could help",
    "asked voice", "barely whisper", "heart pounding", "took deep",
    "felt like", "mind racing", "could feel", "find way",
    "air thick", "voice low", "eyes wide", "stepped forward",
    "shook head", "voice steady", "made way", "could see",
    "eyes widened", "help feel", "first time", "brow furrowed",
    "voice trembling", "pounding chest", "something else", "closed eyes",
    "eyes filled", "felt surge", "eyes narrowed", "shake feeling",
    "dimly lit", "whispered voice", "casting long", "felt sense",
    "felt strange", "mind raced", "barely audible", "could shake",
    "heart racing", "eyes fixed", "heart raced", "world around",
    "young woman", "eyes scanning", "knew would", "feel sense",
    "long shadows", "sense unease", "would never", "thick scent",
    "sense purpose", "run spine", "shadows across", "looked around",
    "replied voice", "one day", "voice laced", "leaned forward",
    "fell silent", "voice filled", "knew could", "across face",
    "voice echoing", "stark contrast", "make sense", "lay ahead",
    "help wonder", "young man", "looked eyes", "took step",
    "hung air", "raised hand", "opened eyes", "sun dipped",
    "leaned back", "one thing", "let go", "gaze fixed",
    "raised eyebrow", "living room", "eyes narrowing", "old man",
    "dim light", "tilted head", "stumbled upon", "swallowed hard",
    "flicker something", "ready face", "deep within", "looked like",
    "something far", "could hear", "never seen", "growing sense",
    "smile playing", "strange sense", "began voice", "let us",
    "ever seen", "almost imperceptible", "constant reminder",
    "dipped horizon", "unlike anything", "also knew", "playing lips",
    "maybe maybe", "sound like", "felt chill", "long time",
    "voice soft", "filled air", "horizon casting", "years ago",
    "voice tinged", "fluorescent lights", "back chair", "glimmer hope",
    "felt weight", "beneath feet", "voice firm", "one one",
    "leaving behind", "days turned", "heart pounded", "around room",
    "would find", "mixture fear", "caught eye", "dark lord",
    "voice dripping", "stepped closer", "air crackled", "fabric reality",
    "caught throat", "held breath", "face whatever", "face etched",
    "figure emerged", "nodded eyes", "trembling slightly", "air around",
    "chill run", "spreading across", "turned weeks", "sense dread",
    "new world", "eyes darting", "sense peace", "mind reeling",
    "eyes gleaming", "left behind", "something akin", "trying make",
    "one would", "opened mouth", "said eyes", "voice echoed",
    "face pale", "voice like", "way back", "could quite",
    "anything ever", "like eternity", "one could", "long since",
    "eyes locked", "breath caught", "beneath surface", "walked away",
    "echoed mind", "beacon hope", "casting warm", "let get",
    "step forward", "chapter seven", "make sure", "carefully constructed",
    "turned face", "spoke voice", "faces etched", "eyes widening",
    "forward eyes", "trying keep", "cleared throat", "could find",
    "turned back", "let see", "felt shiver", "called voice",
    "world outside", "next morning", "elara felt", "felt different",
    "hesitated moment", "felt familiar", "shiver run", "man named",
    "across room",
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_slop_ngrams(text: str, word_count: int) -> float:
    """Count slop trigram + bigram occurrences, normalize to 0-1.

    Trigrams are weighted more heavily than bigrams since they're more
    diagnostic (less likely to occur in legitimate prose).
    """
    text_lower = text.lower()

    trigram_count = sum(1 for t in SLOP_TRIGRAMS if t in text_lower)
    bigram_count = sum(1 for b in SLOP_BIGRAMS if b in text_lower)

    # Trigrams: 3 words each, very diagnostic. Bigrams: 2 words, more common
    # in natural prose, so weight them lower.
    per_k = word_count / 1000
    trigram_density = trigram_count / max(per_k, 1)
    bigram_density = bigram_count / max(per_k, 1)

    # Combined: trigrams 70%, bigrams 30%
    combined_density = (trigram_density * 0.7) + (bigram_density * 0.3)

    if combined_density <= 2:
        return 0.0
    elif combined_density >= 10:
        return 1.0
    return (combined_density - 2) / 8
