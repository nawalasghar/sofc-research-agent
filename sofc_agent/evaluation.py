"""Scoring and summary for the agent evaluation (``eval/run_eval.py`` drives it).

Per question it records which tools the model chose, whether that matches the tool the
question was written for, whether the answer cites a tool result, whether the answer is
right, and how long it took.

"Right" is an automatic check, not a human grade:

* ``answer`` questions pass when every ``;``-separated keyword group in
  ``expected_keywords`` has at least one ``|``-separated alternative in the answer, and no
  alternative of a ``!``-prefixed (forbidden) group appears;
* a numeric alternative is the reference value and matches any number in the answer that
  reports it at the answer's own precision (5.0574 matches "5.06" and "5.1", not "5" or
  "5.03"); a word alternative matches as a whole word ("LN-SDC" not inside "LNK-SDC");
* ``decline`` questions pass when the agent says the tools do not have the answer and
  claims no value of its own (restating the question's numbers is fine; a number no tool
  produced fails it);
* on a two-tool question that includes ``search_literature``, the answer must also cite a
  passage, because keywords cannot check the explanatory half.

Keywords are matched against the answer body with citation brackets removed, never the
appended Sources list: a source line such as "Arrhenius fit: Ea = 0.63 eV" would
otherwise let a wrong number in the model's own text pass.
"""
from __future__ import annotations

import csv
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .agent import _BRACKETED, AgentResult, says_not_found
from .numbers import normalize, number_tokens, reports, to_float

RESULT_FIELDS = [
    "id",
    "category",
    "expected_tools",
    "expected_behavior",
    "tools_called",
    "first_tool",
    "tool_match",
    "first_tool_match",
    "extra_tools",
    "n_tool_calls",
    "cites_source",
    "n_cited_sources",
    "n_invalid_citations",
    "not_found",
    "answer_correct",
    "ungrounded_numbers",
    "nudged",
    "repaired",
    "stop_reason",
    "llm_calls",
    "prompt_tokens",
    "completion_tokens",
    "time_s",
    "question",
    "answer",
    "error",
]

QUESTION_FIELDS = ["id", "category", "expected_tools", "expected_behavior", "question", "expected_keywords"]


def yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def load_questions(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [field for field in QUESTION_FIELDS if rows and field not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    ids = [row["id"] for row in rows]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate question ids in {path}: {duplicates}")
    for row in rows:
        if row["expected_behavior"] not in {"answer", "decline"}:
            raise ValueError(f"{row['id']}: expected_behavior must be 'answer' or 'decline'")
        if row["category"] not in {"single", "two_tool"}:
            raise ValueError(f"{row['id']}: category must be 'single' or 'two_tool'")
    return rows


def expected_tools(row: dict) -> list[str]:
    return [t.strip() for t in row.get("expected_tools", "").split(";") if t.strip()]


_NUMERIC = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _present(alternative: str, body: str, tokens: list[str]) -> bool:
    if _NUMERIC.fullmatch(alternative):
        reference = float(alternative)
        for token in tokens:
            if re.search(r"[xX^eE]", token):  # scientific notation: compare by value
                try:
                    if abs(to_float(token) - reference) <= 5e-3 * abs(reference):
                        return True
                except ValueError:
                    continue
            elif reports(token, reference):
                return True
        return False
    # A plain word matches as a word start ("electron" in "electrons", "coat" in
    # "coating", but "stable" not in "unstable"); anything with digits or symbols
    # ("LN-SDC", "Ia-3", "(Li/Na)2CO3") must match whole.
    tail = "" if re.fullmatch(r"[A-Za-z ]+", alternative) else r"(?![A-Za-z0-9])"
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(alternative)}{tail}"
    return re.search(pattern, body, re.I) is not None


def _groups(spec: str) -> list[tuple[bool, list[str], str | None]]:
    """(forbidden, alternatives, subject) per ';' group. '!a|b@CeO2' forbids a and b in
    the clauses that name CeO2 only, so a remark about another polymorph does not count."""
    groups = []
    for group in (spec or "").split(";"):
        group = group.strip()
        if not group:
            continue
        forbidden = group.startswith("!")
        group = group.lstrip("!")
        subject = None
        if "@" in group:
            group, subject = (part.strip() for part in group.rsplit("@", 1))
        alternatives = [a.strip() for a in group.split("|") if a.strip()]
        if alternatives:
            groups.append((forbidden, alternatives, subject or None))
    return groups


def _clauses(body: str, subject: str | None) -> list[str]:
    if not subject:
        return [body]
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(subject)}(?![A-Za-z0-9])"
    return [c for c in re.split(r"(?<=[.!?])\s+|;\s*|\n+", body) if re.search(pattern, c, re.I)]


def _hit(alternatives: list[str], text: str) -> bool:
    tokens = number_tokens(text)
    return any(_present(a, text, tokens) for a in alternatives)


def keyword_match(text: str, spec: str) -> bool | None:
    """All positive ';' groups must match (any '|' alternative will do) and no alternative
    of a '!' group may appear. Returns None when there is nothing to check."""
    groups = _groups(spec)
    if not groups:
        return None
    body = normalize(_BRACKETED.sub(" ", text or ""))
    for forbidden, alternatives, subject in groups:
        if forbidden:
            if any(_hit(alternatives, clause) for clause in _clauses(body, subject)):
                return False
        elif not _hit(alternatives, body):
            return False
    return True


def _decline_correct(question: dict, result: AgentResult) -> bool:
    """A decline question is answered well when the agent says the tools lack the answer,
    claims no value of its own, and says nothing the question forbids. Optional positive
    keywords name other acceptable ways of saying so (e.g. "not indexed")."""
    groups = _groups(question.get("expected_keywords", ""))
    body = normalize(_BRACKETED.sub(" ", result.answer_body or ""))
    for forbidden, alternatives, subject in groups:
        if forbidden and any(_hit(alternatives, clause) for clause in _clauses(body, subject)):
            return False
    if result.ungrounded_numbers:  # a number no tool produced: a guess, not a decline
        return False
    positives = [alternatives for forbidden, alternatives, _ in groups if not forbidden]
    return bool(
        result.not_found
        or says_not_found(result.answer_body, known=question.get("question", ""))
        or (positives and all(_hit(alternatives, body) for alternatives in positives))
    )


def score(question: dict, result: AgentResult) -> dict:
    """One results.csv row for a finished question."""
    expected = expected_tools(question)
    used = result.tools_used
    first = used[0] if used else ""
    if question["expected_behavior"] == "decline":
        correct = _decline_correct(question, result)
    else:
        matched = keyword_match(result.answer_body, question.get("expected_keywords", ""))
        correct = (not result.not_found) and matched is not False
        if question["category"] == "two_tool" and "search_literature" in expected:
            # Keywords cannot check the explanatory half, so it must at least come from a
            # cited passage; otherwise "0.45 eV [ref11]" alone would pass as a full answer.
            correct = correct and any(s.tool == "search_literature" for s in result.cited_sources)
    return {
        "id": question["id"],
        "category": question["category"],
        "expected_tools": ";".join(expected),
        "expected_behavior": question["expected_behavior"],
        "tools_called": ";".join(used),
        "first_tool": first,
        "tool_match": yes_no(set(expected) <= set(used)),
        "first_tool_match": yes_no(first in expected),
        "extra_tools": ";".join(t for t in used if t not in expected),
        "n_tool_calls": len(result.tool_calls),
        "cites_source": yes_no(result.cited),
        "n_cited_sources": len(result.cited_sources),
        "n_invalid_citations": len(result.invalid_citations),
        "not_found": yes_no(result.not_found),
        "answer_correct": yes_no(correct),
        "ungrounded_numbers": ";".join(result.ungrounded_numbers),
        "nudged": yes_no(result.nudged),
        "repaired": yes_no(result.repaired),
        "stop_reason": result.stop_reason,
        "llm_calls": result.llm_calls,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "time_s": f"{result.elapsed_s:.1f}",
        "question": question["question"],
        "answer": result.answer,
        "error": "",
    }


def error_row(question: dict, message: str, elapsed_s: float) -> dict:
    """A results.csv row for a question the agent could not finish (backend failure)."""
    row = {field: "" for field in RESULT_FIELDS}
    row.update({
        "id": question["id"],
        "category": question["category"],
        "expected_tools": ";".join(expected_tools(question)),
        "expected_behavior": question["expected_behavior"],
        "time_s": f"{elapsed_s:.1f}",
        "question": question["question"],
        "error": message,
    })
    return row


# ------------------------------------------------------------------------ summary

def _rate(rows: list[dict], field: str, value: str = "yes") -> tuple[int, int]:
    return sum(1 for r in rows if r.get(field) == value), len(rows)


def _pct(hit: int, total: int) -> str:
    return f"{hit}/{total} ({100 * hit / total:.0f}%)" if total else "n/a"


def summarize(rows: Iterable[dict]) -> dict[str, Any]:
    """Aggregate statistics over results.csv rows."""
    rows = list(rows)
    done = [r for r in rows if not r.get("error")]
    single = [r for r in done if r["category"] == "single"]
    two = [r for r in done if r["category"] == "two_tool"]
    answerable = [r for r in done if r["expected_behavior"] == "answer"]
    declines = [r for r in done if r["expected_behavior"] == "decline"]
    answered = [r for r in done if r.get("not_found") == "no"]
    times = [float(r["time_s"]) for r in done if r.get("time_s")]

    per_tool: dict[str, dict] = {}
    for tool in ("search_literature", "query_dataset", "arrhenius", "materials_project"):
        group = [r for r in single if r["expected_tools"] == tool]
        group_answered = [r for r in group if r.get("not_found") == "no"]
        per_tool[tool] = {
            "n": len(group),
            "tool_match": _rate(group, "tool_match"),
            "first_tool_match": _rate(group, "first_tool_match"),
            "correct": _rate(group, "answer_correct"),
            "cited_when_answered": _rate(group_answered, "cites_source"),
            "mean_time_s": statistics.mean(float(r["time_s"]) for r in group) if group else None,
        }

    return {
        "n_questions": len(rows),
        "n_completed": len(done),
        "n_errors": len(rows) - len(done),
        "single_tool_match": _rate(single, "tool_match"),
        "single_first_tool_match": _rate(single, "first_tool_match"),
        "single_exact_tool": (sum(1 for r in single if r["tool_match"] == "yes" and not r["extra_tools"]), len(single)),
        "two_tool_match": _rate(two, "tool_match"),
        "overall_tool_match": _rate(done, "tool_match"),
        "cited_when_answered": _rate(answered, "cites_source"),
        "cited_first_pass": (
            sum(1 for r in answered if r["cites_source"] == "yes" and r["repaired"] == "no"), len(answered)
        ),
        "answerable_correct": _rate(answerable, "answer_correct"),
        "declines_correct": _rate(declines, "answer_correct"),
        "overall_correct": _rate(done, "answer_correct"),
        "false_declines": _rate(answerable, "not_found"),
        "invalid_citations": sum(int(r["n_invalid_citations"] or 0) for r in done),
        "answers_with_ungrounded_numbers": (sum(1 for r in answered if r["ungrounded_numbers"]), len(answered)),
        "nudged": _rate(done, "nudged"),
        "repaired": _rate(done, "repaired"),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in done)),
        "time_mean_s": statistics.mean(times) if times else None,
        "time_median_s": statistics.median(times) if times else None,
        "time_min_s": min(times) if times else None,
        "time_max_s": max(times) if times else None,
        "time_total_s": sum(times),
        "mean_llm_calls": statistics.mean(int(r["llm_calls"]) for r in done) if done else None,
        "mean_tool_calls": statistics.mean(int(r["n_tool_calls"]) for r in done) if done else None,
        "per_tool": per_tool,
    }


def render_summary(stats: dict, rows: list[dict], meta: dict) -> str:
    """Markdown summary for eval/summary.md."""
    s = stats
    lines = [
        "# Evaluation summary",
        "",
        f"- Model: `{meta.get('model')}` via Ollama {meta.get('ollama_version', '')}".rstrip(),
        f"- Hardware: {meta.get('hardware', 'not recorded')}",
        f"- Run finished: {meta.get('finished', 'not recorded')}",
        f"- Questions: {meta.get('question_set_size', s['n_questions'])} in the set; "
        f"{s['n_completed']} completed, {s['n_errors']} ended in an error"
        + (f", {len(meta['not_run'])} not run yet ({', '.join(meta['not_run'])})" if meta.get("not_run") else "")
        + f"; at most {meta.get('max_tool_calls')} tool calls per question",
        f"- Warm-up (model + embedding load, not counted below): {meta.get('warmup_s', 'n/a')} s",
        "",
        "## Headline numbers",
        "",
        "| Metric | Result |",
        "| --- | --- |",
        f"| Expected tool called, single-tool questions | {_pct(*s['single_tool_match'])} |",
        f"| ...and it was the first tool called | {_pct(*s['single_first_tool_match'])} |",
        f"| ...and no other tool was called | {_pct(*s['single_exact_tool'])} |",
        f"| Both expected tools called, two-tool questions | {_pct(*s['two_tool_match'])} |",
        f"| Answer cites a tool result (answered questions) | {_pct(*s['cited_when_answered'])} |",
        f"| ...cited without needing the repair prompt | {_pct(*s['cited_first_pass'])} |",
        f"| Answer correct, answerable questions (keyword check) | {_pct(*s['answerable_correct'])} |",
        f"| Correctly declined, unanswerable questions | {_pct(*s['declines_correct'])} |",
        f"| Wrongly declined, answerable questions | {_pct(*s['false_declines'])} |",
        f"| Invented citation tags caught | {s['invalid_citations']} |",
        f"| Answers with numbers not found in any tool output | {_pct(*s['answers_with_ungrounded_numbers'])} |",
        f"| Time per question: mean / median | {s['time_mean_s']:.1f} s / {s['time_median_s']:.1f} s |"
        if s["time_mean_s"] is not None else "| Time per question | n/a |",
        f"| Time per question: min / max | {s['time_min_s']:.1f} s / {s['time_max_s']:.1f} s |"
        if s["time_min_s"] is not None else None,
        f"| Total time | {s['time_total_s'] / 60:.1f} min |",
        f"| Model calls / tool calls per question (mean) | {s['mean_llm_calls']:.1f} / {s['mean_tool_calls']:.1f} |"
        if s["mean_llm_calls"] is not None else None,
        "",
        "## By expected tool (single-tool questions)",
        "",
        "| Tool | Questions | Tool matched | Called first | Correct | Cited when answered | Mean time |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for tool, t in s["per_tool"].items():
        mean = f"{t['mean_time_s']:.1f} s" if t["mean_time_s"] is not None else "n/a"
        lines.append(
            f"| `{tool}` | {t['n']} | {_pct(*t['tool_match'])} | {_pct(*t['first_tool_match'])} | "
            f"{_pct(*t['correct'])} | {_pct(*t['cited_when_answered'])} | {mean} |"
        )
    lines += [
        "",
        "## Per question",
        "",
        "| ID | Expected | Called | Match | Cites | Correct | Time |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['id']} | {r['expected_tools']} | error: {r['error'][:60]} | | | | {r['time_s']} s |")
            continue
        behavior = " (decline)" if r["expected_behavior"] == "decline" else ""
        lines.append(
            f"| {r['id']} | {r['expected_tools'].replace(';', ' + ')}{behavior} | "
            f"{r['tools_called'].replace(';', ' + ') or 'none'} | {r['tool_match']} | "
            f"{r['cites_source']} | {r['answer_correct']} | {r['time_s']} s |"
        )
    stop = ", ".join(f"{k}: {v}" for k, v in sorted(s["stop_reasons"].items()))
    lines += [
        "",
        f"Stop reasons: {stop}. Nudged (no tool on first try): {_pct(*s['nudged'])}. "
        f"Repair prompt used: {_pct(*s['repaired'])}.",
        "",
        "Correctness is an automatic keyword check against the answer text (see "
        "`expected_keywords` in `questions.csv`), not a human grade; it is lenient for the "
        "explanatory literature questions. Full answers are in `results.csv`, and every tool "
        "call with its arguments is in `traces.jsonl`.",
        "",
    ]
    return "\n".join(line for line in lines if line is not None)
