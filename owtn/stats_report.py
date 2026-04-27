"""End-of-run statistics report.

Reads artifacts from a completed (or in-progress) run directory — LLM call
YAML logs, programs.sqlite, champions/, tournament.json — and emits a
structured summary appended to ``evolution_run.log`` plus ``stats.json``.

Can be invoked directly on any historical run:
    uv run python -m owtn.stats_report results/run_<ts>/stage_1
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _safe_cost(c: dict) -> float:
    """Read `cost` from a call record, treating NaN/missing as 0.

    A failed API call can write ``.nan`` to its cost field (e.g. when tokens
    are 0 and the pricing lookup divides by zero). One such call was
    poisoning sums via ``nan`` propagation. Treat as 0 — the failure is
    already visible in the zero-token row.
    """
    v = c.get("cost", 0.0)
    try:
        if v is None or math.isnan(float(v)):
            return 0.0
    except Exception:
        return 0.0
    return float(v)

from owtn.evaluation.models import DIMENSION_NAMES

logger = logging.getLogger(__name__)

_VOTE_RE = re.compile(
    r"\b(novelty|grip|tension_architecture|emotional_depth|thematic_resonance|"
    r"concept_coherence|generative_fertility|scope_calibration|indelibility)"
    r"\s*=\s*['\"]([a-z_]+)['\"]"
)


def _load_calls(run_dir: Path) -> list[dict]:
    """Load every LLM call YAML under <run_dir>/llm/<model>/*.yaml."""
    llm_root = run_dir / "llm"
    if not llm_root.is_dir():
        return []
    calls = []
    for path in sorted(llm_root.rglob("*.yaml")):
        try:
            rec = yaml.safe_load(path.read_text())
        except Exception as e:
            logger.warning("Skipping %s: %s", path, e)
            continue
        if isinstance(rec, dict):
            calls.append(rec)
    return calls


def _premise_fingerprint(user_msg: str) -> tuple[str, str]:
    """Extract the first two Premise: lines from a pairwise judge user message.

    Returns (A-premise-head, B-premise-head), each truncated to 120 chars.
    Used to identify forward/reverse ordering of the same match.
    """
    premises = re.findall(r"Premise:\s*(.{0,120})", user_msg)
    if len(premises) < 2:
        return ("", "")
    return (premises[0].strip(), premises[1].strip())


def _parse_votes(output: str) -> dict[str, str]:
    """Extract the 9 dimension verdicts from a judge's raw output string."""
    votes = {}
    for m in _VOTE_RE.finditer(output):
        dim, val = m.group(1), m.group(2)
        if dim not in votes and val in _VALID_VOTES:
            votes[dim] = val
    return votes


_VALID_VOTES = {
    "a_narrow", "a_clear", "a_decisive",
    "b_narrow", "b_clear", "b_decisive",
    "tie",
}

_MAG_LABELS = {"narrow", "clear", "decisive"}


def _vote_side(vote: str) -> str:
    if vote == "tie":
        return "tie"
    return vote.split("_", 1)[0]


def _vote_magnitude(vote: str) -> str:
    if vote == "tie":
        return "tie"
    return vote.split("_", 1)[1]


def _flip_side(vote: str) -> str:
    if vote == "tie":
        return "tie"
    side, mag = vote.split("_", 1)
    return ("b_" if side == "a" else "a_") + mag


# ---------------------------------------------------------------------------
# Cost & model tables
# ---------------------------------------------------------------------------

def _compute_model_table(calls: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for c in calls:
        by_model[c.get("model", "?")].append(c)
    rows = []
    for model, recs in sorted(by_model.items()):
        in_tok = sum(r.get("tokens", {}).get("input", 0) for r in recs)
        out_tok = sum(r.get("tokens", {}).get("output", 0) for r in recs)
        cache_read = sum(r.get("tokens", {}).get("cache_read", 0) for r in recs)
        cost = sum(_safe_cost(r) for r in recs)
        durations = [r.get("duration_s", 0.0) for r in recs if r.get("duration_s") is not None]
        p50 = statistics.median(durations) if durations else 0.0
        p95 = _percentile(durations, 95) if durations else 0.0
        cache_hit_rate = (cache_read / in_tok) if in_tok else 0.0
        rows.append({
            "model": model,
            "calls": len(recs),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cache_read_tokens": cache_read,
            "cache_hit_rate": round(cache_hit_rate, 3),
            "cost": round(cost, 4),
            "latency_p50": round(p50, 2),
            "latency_p95": round(p95, 2),
        })
    return rows


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _compute_cost_by_role(calls: list[dict]) -> dict[str, dict]:
    by_role: dict[str, list[dict]] = defaultdict(list)
    for c in calls:
        by_role[c.get("role", "unknown")].append(c)
    out = {}
    for role, recs in sorted(by_role.items()):
        out[role] = {
            "calls": len(recs),
            "cost": round(sum(_safe_cost(r) for r in recs), 4),
            "input_tokens": sum(r.get("tokens", {}).get("input", 0) for r in recs),
            "output_tokens": sum(r.get("tokens", {}).get("output", 0) for r in recs),
        }
    return out


# ---------------------------------------------------------------------------
# Judge diagnostics
# ---------------------------------------------------------------------------

def _pair_judge_calls(judge_calls: list[dict]) -> list[tuple[dict, dict]]:
    """Group pairwise_judge calls into (forward, reverse) pairs.

    Pairing uses the (A-premise, B-premise) fingerprint: two calls with
    swapped premises form a pair. Unpaired calls are dropped.
    """
    by_pair: dict[frozenset, list[dict]] = defaultdict(list)
    for c in judge_calls:
        fp = _premise_fingerprint(c.get("user_msg", ""))
        if not fp[0] or not fp[1]:
            continue
        by_pair[frozenset(fp)].append(c)
    pairs = []
    for members in by_pair.values():
        if len(members) != 2:
            continue
        c0, c1 = members
        fp0 = _premise_fingerprint(c0.get("user_msg", ""))
        fp1 = _premise_fingerprint(c1.get("user_msg", ""))
        # Declare the lex-smaller A-premise as "forward"; flips are relative.
        if fp0[0] <= fp1[0]:
            pairs.append((c0, c1))
        else:
            pairs.append((c1, c0))
    return pairs


def _resolve_pair(
    fwd_votes: dict[str, str],
    rev_votes: dict[str, str],
) -> dict[str, str]:
    """Return per-dim resolved verdict (same side in both orderings, else tie)."""
    resolved = {}
    for dim in DIMENSION_NAMES:
        fv = fwd_votes.get(dim, "tie")
        rv = rev_votes.get(dim, "tie")
        fwd_side = _vote_side(fv)
        rev_side = _vote_side(_flip_side(rv))
        if fwd_side == "tie" or rev_side == "tie" or fwd_side != rev_side:
            resolved[dim] = "tie"
        else:
            resolved[dim] = fwd_side
    return resolved


def _compute_judge_stats(calls: list[dict]) -> dict[str, Any]:
    judge_calls = [c for c in calls if c.get("role") == "pairwise_judge"]
    by_judge: dict[str, list[dict]] = defaultdict(list)
    for c in judge_calls:
        jid = c.get("judge_id")
        if jid:
            by_judge[jid].append(c)

    per_judge: dict[str, dict] = {}
    per_judge_pair_resolved: dict[str, dict[frozenset, dict[str, str]]] = {}

    for judge_id, recs in sorted(by_judge.items()):
        pairs = _pair_judge_calls(recs)
        dim_flipped = 0
        dim_consistent = 0
        dim_tie = 0
        mag_counts = Counter()
        side_counts = Counter()
        per_dim_side = defaultdict(Counter)
        resolved_by_pair: dict[frozenset, dict[str, str]] = {}

        for fwd, rev in pairs:
            fwd_votes = _parse_votes(fwd.get("output", ""))
            rev_votes = _parse_votes(rev.get("output", ""))
            fp = frozenset(_premise_fingerprint(fwd.get("user_msg", "")))

            for dim in DIMENSION_NAMES:
                fv = fwd_votes.get(dim, "tie")
                rv = rev_votes.get(dim, "tie")
                fwd_side = _vote_side(fv)
                rev_side_flipped = _vote_side(_flip_side(rv))

                if fwd_side == "tie" or rev_side_flipped == "tie":
                    dim_tie += 1
                elif fwd_side == rev_side_flipped:
                    dim_consistent += 1
                else:
                    dim_flipped += 1

            # Magnitude distribution from forward ordering only (avoid double-count).
            for v in fwd_votes.values():
                mag_counts[_vote_magnitude(v)] += 1
                side_counts[_vote_side(v)] += 1
                # Per-dim tracking done below.

            for dim in DIMENSION_NAMES:
                per_dim_side[dim][_vote_side(fwd_votes.get(dim, "tie"))] += 1

            resolved_by_pair[fp] = _resolve_pair(fwd_votes, rev_votes)

        total_dims = dim_consistent + dim_flipped + dim_tie
        decisive_dims = dim_consistent + dim_flipped
        per_judge[judge_id] = {
            "pairs": len(pairs),
            "calls": len(recs),
            "dim_consistent": dim_consistent,
            "dim_flipped": dim_flipped,
            "dim_tied_before_resolve": dim_tie,
            "flip_rate": round(dim_flipped / decisive_dims, 3) if decisive_dims else None,
            "tie_rate_pre_resolve": round(dim_tie / total_dims, 3) if total_dims else None,
            "magnitude_dist": dict(mag_counts),
            "side_dist": dict(side_counts),
            "per_dim_side": {d: dict(per_dim_side[d]) for d in DIMENSION_NAMES},
        }
        per_judge_pair_resolved[judge_id] = resolved_by_pair

    # Leave-one-out panel agreement: for each pair seen by >=3 judges, for each
    # dim, check whether judge X's resolved side matches the majority of the
    # other judges. Ties on either side count as abstention (not counted
    # toward agreement denominator).
    all_pair_fps = set()
    for m in per_judge_pair_resolved.values():
        all_pair_fps.update(m.keys())

    loo_num: dict[str, int] = defaultdict(int)
    loo_den: dict[str, int] = defaultdict(int)
    for fp in all_pair_fps:
        judges_seen = [j for j, m in per_judge_pair_resolved.items() if fp in m]
        if len(judges_seen) < 3:
            continue
        for dim in DIMENSION_NAMES:
            sides = {j: per_judge_pair_resolved[j][fp].get(dim, "tie") for j in judges_seen}
            for j in judges_seen:
                if sides[j] == "tie":
                    continue
                others = [s for k, s in sides.items() if k != j and s != "tie"]
                if not others:
                    continue
                maj = Counter(others).most_common(1)[0][0]
                loo_den[j] += 1
                if sides[j] == maj:
                    loo_num[j] += 1

    for j in per_judge:
        den = loo_den.get(j, 0)
        per_judge[j]["loo_agreement"] = (
            round(loo_num.get(j, 0) / den, 3) if den else None
        )
        per_judge[j]["loo_sample_size"] = den

    return per_judge


# ---------------------------------------------------------------------------
# Pairwise-tournament aggregate stats (across the run, not post-evolution Swiss)
# ---------------------------------------------------------------------------

def _compute_pairwise_stats(calls: list[dict]) -> dict[str, Any]:
    judge_calls = [c for c in calls if c.get("role") == "pairwise_judge"]
    # Group by match fingerprint → all 2*judges calls for one match.
    by_match: dict[frozenset, list[dict]] = defaultdict(list)
    for c in judge_calls:
        fp = _premise_fingerprint(c.get("user_msg", ""))
        if fp[0] and fp[1]:
            by_match[frozenset(fp)].append(c)

    total_matches = 0
    resolved_ties = 0
    winner_a = 0
    winner_b = 0
    per_judge_count = Counter()
    for fp, members in by_match.items():
        # A match is represented twice per judge (fwd + rev).
        judges = {m.get("judge_id") for m in members}
        per_judge_count.update(judges)
        if len(members) < 2:
            continue
        total_matches += 1

    return {
        "total_matches": total_matches,
        "calls_per_judge": dict(per_judge_count),
    }


# ---------------------------------------------------------------------------
# Evolution / database stats
# ---------------------------------------------------------------------------

def _compute_evolution_stats(run_dir: Path) -> dict[str, Any]:
    db_path = run_dir / "programs.sqlite"
    if not db_path.exists():
        return {"error": "programs.sqlite not found"}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM programs")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM programs WHERE correct = 1")
    correct = cur.fetchone()[0]
    cur.execute("SELECT MAX(generation), MIN(generation) FROM programs")
    max_gen, min_gen = cur.fetchone()

    cur.execute(
        """SELECT generation,
                  COUNT(*) as n,
                  AVG(combined_score) as mean_score,
                  MAX(combined_score) as max_score
             FROM programs
            WHERE correct = 1
            GROUP BY generation
            ORDER BY generation"""
    )
    per_gen = [
        {"generation": r["generation"], "n": r["n"],
         "mean_score": round(r["mean_score"] or 0.0, 4),
         "max_score": round(r["max_score"] or 0.0, 4)}
        for r in cur.fetchall()
    ]

    cur.execute(
        """SELECT island_idx, COUNT(*) as n
             FROM programs WHERE correct = 1
            GROUP BY island_idx ORDER BY island_idx"""
    )
    per_island = [
        {"island": r["island_idx"], "n": r["n"]}
        for r in cur.fetchall()
    ]

    # Parent reuse — sanity check on power-law selection.
    cur.execute(
        """SELECT parent_id, COUNT(*) as c
             FROM programs WHERE parent_id IS NOT NULL AND parent_id != ''
            GROUP BY parent_id ORDER BY c DESC LIMIT 10"""
    )
    top_parents = [
        {"parent_id": r["parent_id"][:8], "children": r["c"]}
        for r in cur.fetchall()
    ]

    # Operator distribution. patch_type is the operator name (collision,
    # anti_premise, ...); patch_name is the model's chosen name for the
    # concept (e.g. "the_sampled_token") — we want the former.
    cur.execute("SELECT metadata FROM programs WHERE metadata IS NOT NULL")
    op_counts = Counter()
    op_success = Counter()
    cur2 = conn.cursor()
    for row in cur.fetchall():
        try:
            meta = json.loads(row["metadata"] or "{}")
        except Exception:
            continue
        op = meta.get("patch_type") or "unknown"
        op_counts[op] += 1

    cur2.execute(
        """SELECT metadata, combined_score, correct FROM programs
            WHERE correct = 1"""
    )
    for row in cur2.fetchall():
        try:
            meta = json.loads(row["metadata"] or "{}")
        except Exception:
            continue
        op = meta.get("patch_type") or "unknown"
        if (row["combined_score"] or 0.0) >= 0.5:
            op_success[op] += 1

    operators = [
        {"operator": op,
         "invocations": op_counts[op],
         "promoted": op_success.get(op, 0),
         "promotion_rate": round(op_success.get(op, 0) / op_counts[op], 3)
                           if op_counts[op] else 0.0}
        for op in sorted(op_counts, key=lambda x: -op_counts[x])
    ]

    # Rejection bookkeeping — validation failures (correct=0).
    cur.execute(
        """SELECT COUNT(*) FROM programs WHERE correct = 0"""
    )
    rejected = cur.fetchone()[0]

    conn.close()

    return {
        "total_programs": total,
        "correct_programs": correct,
        "rejected_programs": rejected,
        "rejection_rate": round(rejected / total, 3) if total else None,
        "generations_covered": [min_gen, max_gen] if max_gen is not None else [],
        "per_generation": per_gen,
        "per_island": per_island,
        "top_parents": top_parents,
        "operators": operators,
    }


# ---------------------------------------------------------------------------
# Archive stats (fitness-based, not MAP-Elites)
# ---------------------------------------------------------------------------

def _compute_archive_stats(run_dir: Path) -> dict[str, Any]:
    db_path = run_dir / "programs.sqlite"
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute(
            """SELECT p.generation, p.combined_score
                 FROM archive a
                 JOIN programs p ON a.program_id = p.id
                ORDER BY p.combined_score DESC"""
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return {}
    conn.close()
    if not rows:
        return {"size": 0}
    scores = [r["combined_score"] for r in rows]
    gens = [r["generation"] for r in rows]
    gen_counts = Counter(gens)
    return {
        "size": len(rows),
        "max_score": round(max(scores), 4),
        "mean_score": round(sum(scores) / len(scores), 4),
        "min_score": round(min(scores), 4),
        "entries_per_generation": dict(sorted(gen_counts.items())),
    }


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

def _compute_run_summary(calls: list[dict]) -> dict[str, Any]:
    if not calls:
        return {}
    timestamps = []
    for c in calls:
        ts = c.get("timestamp")
        if not ts:
            continue
        try:
            timestamps.append(datetime.fromisoformat(ts))
        except Exception:
            continue
    duration_s = 0.0
    if timestamps:
        duration_s = (max(timestamps) - min(timestamps)).total_seconds()

    total_cost = sum(_safe_cost(c) for c in calls)
    total_input = sum(c.get("tokens", {}).get("input", 0) for c in calls)
    total_output = sum(c.get("tokens", {}).get("output", 0) for c in calls)
    total_cache = sum(c.get("tokens", {}).get("cache_read", 0) for c in calls)

    return {
        "total_calls": len(calls),
        "total_cost": round(total_cost, 4),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cache_read_tokens": total_cache,
        "cache_hit_rate": round(total_cache / total_input, 3) if total_input else 0.0,
        "wall_duration_s": round(duration_s, 1),
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def compute_stats(run_dir: Path) -> dict[str, Any]:
    calls = _load_calls(run_dir)
    return {
        "run_dir": str(run_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_summary": _compute_run_summary(calls),
        "cost_by_role": _compute_cost_by_role(calls),
        "per_model": _compute_model_table(calls),
        "judges": _compute_judge_stats(calls),
        "pairwise": _compute_pairwise_stats(calls),
        "evolution": _compute_evolution_stats(run_dir),
        "archive": _compute_archive_stats(run_dir),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    """Render rows as a space-padded text table. cols = [(header, key), ...]."""
    if not rows:
        return "  (no data)"
    headers = [h for h, _ in cols]
    widths = [len(h) for h in headers]
    str_rows = []
    for r in rows:
        sr = []
        for i, (_, k) in enumerate(cols):
            v = r.get(k, "")
            s = f"{v:.4f}" if isinstance(v, float) else str(v)
            widths[i] = max(widths[i], len(s))
            sr.append(s)
        str_rows.append(sr)
    line = "  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  " + "  ".join("-" * w for w in widths)
    body = ["  " + "  ".join(sr[i].ljust(widths[i]) for i in range(len(cols)))
            for sr in str_rows]
    return "\n".join([line, sep, *body])


def format_stats(stats: dict[str, Any]) -> str:
    out = []
    bar = "=" * 78
    out.append(bar)
    out.append("  RUN STATISTICS")
    out.append(f"  {stats.get('run_dir')}")
    out.append(f"  generated: {stats.get('generated_at')}")
    out.append(bar)

    rs = stats.get("run_summary", {})
    out.append("\n[ RUN SUMMARY ]")
    if rs:
        out.append(f"  total LLM calls      : {rs.get('total_calls')}")
        out.append(f"  total cost           : ${rs.get('total_cost'):.4f}")
        out.append(f"  wall duration        : {rs.get('wall_duration_s')}s "
                   f"({rs.get('wall_duration_s', 0) / 60:.1f} min)")
        out.append(f"  input tokens         : {rs.get('total_input_tokens'):,}")
        out.append(f"  output tokens        : {rs.get('total_output_tokens'):,}")
        out.append(f"  cache-read tokens    : {rs.get('total_cache_read_tokens'):,}")
        out.append(f"  cache hit rate       : {rs.get('cache_hit_rate'):.1%}")

    cbr = stats.get("cost_by_role", {})
    if cbr:
        out.append("\n[ COST BY ROLE ]")
        rows = [{"role": r, **v} for r, v in cbr.items()]
        out.append(_fmt_table(
            rows,
            [("role", "role"), ("calls", "calls"), ("cost $", "cost"),
             ("input_tok", "input_tokens"), ("output_tok", "output_tokens")],
        ))

    per_model = stats.get("per_model", [])
    if per_model:
        out.append("\n[ PER-MODEL ]")
        out.append(_fmt_table(
            per_model,
            [("model", "model"), ("calls", "calls"),
             ("input_tok", "input_tokens"), ("output_tok", "output_tokens"),
             ("cost $", "cost"), ("cache%", "cache_hit_rate"),
             ("p50 s", "latency_p50"), ("p95 s", "latency_p95")],
        ))

    judges = stats.get("judges", {})
    if judges:
        out.append("\n[ JUDGES ]")
        rows = []
        for jid, js in judges.items():
            mag = js.get("magnitude_dist", {})
            side = js.get("side_dist", {})
            tot_side = sum(side.values()) or 1
            rows.append({
                "judge": jid,
                "pairs": js.get("pairs"),
                "flip%": f"{(js.get('flip_rate') or 0) * 100:.1f}",
                "tie_pre%": f"{(js.get('tie_rate_pre_resolve') or 0) * 100:.1f}",
                "loo%": (f"{js.get('loo_agreement') * 100:.1f}"
                         if js.get("loo_agreement") is not None else "-"),
                "loo_n": js.get("loo_sample_size") or 0,
                "a%": f"{side.get('a', 0) / tot_side * 100:.0f}",
                "b%": f"{side.get('b', 0) / tot_side * 100:.0f}",
                "tie%": f"{side.get('tie', 0) / tot_side * 100:.0f}",
                "narrow": mag.get("narrow", 0),
                "clear": mag.get("clear", 0),
                "decisive": mag.get("decisive", 0),
            })
        out.append(_fmt_table(
            rows,
            [("judge", "judge"), ("pairs", "pairs"),
             ("flip%", "flip%"), ("tie_pre%", "tie_pre%"),
             ("loo%", "loo%"), ("loo_n", "loo_n"),
             ("a%", "a%"), ("b%", "b%"), ("tie%", "tie%"),
             ("narrow", "narrow"), ("clear", "clear"), ("decisive", "decisive")],
        ))
        out.append("  flip%   = dim-level position flips / decisive dims "
                   "(panel-wide ~45% is industry baseline per Shi et al.)")
        out.append("  loo%    = agreement with majority of other judges on "
                   "resolved per-dim side")
        out.append("  a/b/tie = forward-ordering side distribution — skew "
                   "signals position-bias or self-preference")

    pw = stats.get("pairwise", {})
    if pw:
        out.append("\n[ PAIRWISE TOURNAMENT ]")
        out.append(f"  total matches: {pw.get('total_matches')}")
        cpj = pw.get("calls_per_judge", {})
        if cpj:
            out.append("  calls per judge:")
            for j, n in sorted(cpj.items()):
                out.append(f"    {j:<25} {n}")

    ev = stats.get("evolution", {})
    if ev and "error" not in ev:
        out.append("\n[ EVOLUTION ]")
        out.append(f"  total programs   : {ev.get('total_programs')}")
        out.append(f"  correct / kept   : {ev.get('correct_programs')}")
        out.append(f"  rejected         : {ev.get('rejected_programs')} "
                   f"(rate {ev.get('rejection_rate')})")
        gc = ev.get("generations_covered", [])
        if gc:
            out.append(f"  generations      : {gc[0]} - {gc[1]}")

        per_gen = ev.get("per_generation", [])
        if per_gen:
            out.append("  per-generation (correct programs only):")
            out.append(_fmt_table(
                per_gen,
                [("gen", "generation"), ("n", "n"),
                 ("mean", "mean_score"), ("max", "max_score")],
            ))

        per_island = ev.get("per_island", [])
        if per_island:
            out.append("  per-island:")
            out.append(_fmt_table(
                per_island,
                [("island", "island"), ("n", "n")],
            ))

        ops = ev.get("operators", [])
        if ops:
            out.append("  operators (by invocations):")
            out.append(_fmt_table(
                ops,
                [("operator", "operator"), ("invocations", "invocations"),
                 ("promoted", "promoted"), ("promo_rate", "promotion_rate")],
            ))

        tp = ev.get("top_parents", [])
        if tp:
            out.append("  top parents (reuse sanity for power-law selection):")
            out.append(_fmt_table(
                tp,
                [("parent", "parent_id"), ("children", "children")],
            ))

    arch = stats.get("archive", {})
    if arch:
        out.append("\n[ ARCHIVE ]")
        out.append(f"  size       : {arch.get('size')}")
        if arch.get("size"):
            out.append(f"  score range: {arch.get('min_score')} - {arch.get('max_score')} "
                       f"(mean {arch.get('mean_score')})")
            epg = arch.get("entries_per_generation", {})
            if epg:
                out.append("  entries by entry-generation:")
                for g, n in epg.items():
                    out.append(f"    gen {g}: {n}")

    out.append("\n" + bar)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# File I/O entry point
# ---------------------------------------------------------------------------

def write_stats(run_dir: Path) -> None:
    """Compute, print to evolution_run.log, and write stats.json."""
    run_dir = Path(run_dir)
    stats = compute_stats(run_dir)
    text = format_stats(stats)

    log_path = run_dir / "evolution_run.log"
    try:
        with log_path.open("a") as fh:
            fh.write("\n" + text + "\n")
    except Exception as e:
        logger.warning("Could not append stats to %s: %s", log_path, e)

    json_path = run_dir / "stats.json"
    try:
        json_path.write_text(json.dumps(stats, indent=2, default=str))
    except Exception as e:
        logger.warning("Could not write %s: %s", json_path, e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-of-run statistics report.")
    parser.add_argument("run_dir", help="Path to stage_1 results directory.")
    parser.add_argument("--print", action="store_true",
                        help="Also print the text block to stdout.")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        return 1
    write_stats(run_dir)
    if args.print:
        print(format_stats(compute_stats(run_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
