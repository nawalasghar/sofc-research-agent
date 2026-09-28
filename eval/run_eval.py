"""Run every question in eval/questions.csv through the agent and score it.

    python eval/run_eval.py                  # run; resumes where a previous run stopped
    python eval/run_eval.py --fresh          # start over (old outputs renamed to *.bak)
    python eval/run_eval.py --only q06,q11   # a subset
    python eval/run_eval.py --summary-only   # rebuild summary.md from results.csv

Crash safety: each finished question is appended to results.csv, and its full trace to
traces.jsonl, then flushed and fsynced before the next question starts. A crash loses at
most the question in flight. On restart, finished questions are skipped and questions
that failed with a backend error are retried.

Questions run one at a time on purpose: the timings are only meaningful if nothing else
competes for the CPU or for the Ollama server.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from sofc_agent.agent import Agent, AgentError, OllamaChat, is_infrastructure_error  # noqa: E402
from sofc_agent.evaluation import (  # noqa: E402
    RESULT_FIELDS,
    error_row,
    load_questions,
    render_summary,
    score,
    summarize,
)

QUESTIONS = EVAL_DIR / "questions.csv"
RESULTS = EVAL_DIR / "results.csv"
TRACES = EVAL_DIR / "traces.jsonl"
SUMMARY = EVAL_DIR / "summary.md"
META = EVAL_DIR / "run_meta.json"


def _durable_append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _csv_text(rows: list[dict], header: bool) -> str:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RESULT_FIELDS, lineterminator="\n")
    if header:
        writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def well_formed(row: dict) -> bool:
    """A complete CSV row: no missing trailing columns and no overflow columns.

    A hard kill can leave a torn last line, and a missing final newline can merge two rows
    into one; both fail this check instead of being read as a finished question.
    """
    return None not in row and all(value is not None for value in row.values()) and bool(row.get("id"))


def finished(row: dict) -> bool:
    return well_formed(row) and bool(row["time_s"]) and row["error"] == ""


def read_results() -> list[dict]:
    if not RESULTS.exists() or RESULTS.stat().st_size == 0:
        return []
    with RESULTS.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if list(reader.fieldnames or []) != RESULT_FIELDS:
            raise SystemExit(
                f"{RESULTS.name} has different columns than this version writes; "
                "rerun with --fresh to start a new evaluation."
            )
        return list(reader)


def current_results(rows: list[dict] | None = None) -> dict[str, dict]:
    """The row that counts for each question: its last finished row, or failing that its
    last error row. A rerun therefore replaces an answer only once it has produced one."""
    latest: dict[str, dict] = {}
    for row in read_results() if rows is None else rows:
        if not well_formed(row):
            continue
        previous = latest.get(row["id"])
        if finished(row) or previous is None or not finished(previous):
            latest[row["id"]] = row
    return latest


def _keep_traces(ids: set[str]) -> None:
    """Keep one trace (the last) per finished question; drop traces of unfinished ones."""
    if not TRACES.exists():
        return
    latest: dict[str, str] = {}
    for line in TRACES.read_text(encoding="utf-8").splitlines():
        try:
            trace_id = json.loads(line).get("id")
        except ValueError:
            continue  # torn line
        if trace_id in ids:
            latest[trace_id] = line
    _atomic_write(TRACES, "".join(latest[i] + "\n" for i in sorted(latest)))


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _ollama_info(url: str) -> tuple[str, str]:
    """(Ollama version, where the model runs) - best effort, never fatal."""
    import requests

    version, placement = "", "unknown"
    try:
        version = requests.get(f"{url}/api/version", timeout=10).json().get("version", "")
        models = requests.get(f"{url}/api/ps", timeout=10).json().get("models", [])
        if models:
            placement = "GPU" if models[0].get("size_vram") else "CPU only"
    except Exception:
        pass
    return version, placement


def warm_up(chat: OllamaChat) -> float:
    """Load the model into memory and the embedding model into this process."""
    from sofc_agent.tools import search_literature

    start = time.perf_counter()
    chat([{"role": "user", "content": "Reply with the word OK."}], None)
    search_literature("warm up", k=1)
    return time.perf_counter() - start


def write_summary(questions: list[dict], meta: dict) -> str:
    current = current_results()
    rows = [current[q["id"]] for q in questions if q["id"] in current]
    missing = [q["id"] for q in questions if q["id"] not in current]
    text = render_summary(summarize(rows), rows, {**meta, "question_set_size": len(questions), "not_run": missing})
    _atomic_write(SUMMARY, text)
    return text


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate the SOFC research agent.")
    parser.add_argument("--only", help="comma-separated question ids to run")
    parser.add_argument("--rerun", action="store_true", help="with --only: redo those questions even if finished")
    parser.add_argument("--fresh", action="store_true", help="discard previous results (kept as *.bak)")
    parser.add_argument("--summary-only", action="store_true", help="rebuild summary.md and exit")
    args = parser.parse_args()

    questions = load_questions(QUESTIONS)
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}

    if args.summary_only:
        print(write_summary(questions, meta))
        return 0

    if args.fresh:
        for path in (RESULTS, TRACES, SUMMARY, META):
            if path.exists():
                os.replace(path, path.with_suffix(path.suffix + ".bak"))
        meta = {}

    wanted = {i.strip() for i in args.only.split(",")} if args.only else None
    unknown = sorted((wanted or set()) - {q["id"] for q in questions})
    if unknown:
        print(f"error: unknown question id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    # One row per question: its finished row if any, else its error row. Anything torn by a
    # hard kill is dropped, and the file is rewritten, which also restores a final newline
    # a crash may have cut off. A --rerun keeps the old answer until the new one exists.
    latest = current_results()
    rerun = wanted if (args.rerun and wanted) else set()
    done_ids = {i for i, row in latest.items() if finished(row) and i not in rerun}
    pending = [q for q in questions if q["id"] not in done_ids and (wanted is None or q["id"] in wanted)]
    pending_ids = {q["id"] for q in pending}
    keep = [row for i, row in latest.items() if finished(row) or i not in pending_ids]
    if RESULTS.exists():
        _atomic_write(RESULTS, _csv_text(keep, header=True))
    _keep_traces({i for i, row in latest.items() if finished(row)})

    if not pending:
        print("Nothing to run: every selected question already has a result.")
        print(write_summary(questions, meta))
        return 0

    chat = OllamaChat()
    agent = Agent(chat=chat)
    print(f"Warming up {chat.model} ...", flush=True)
    try:
        warmup_s = warm_up(chat)
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    version, placement = _ollama_info(chat.url)
    meta.update({
        "model": chat.model,
        "ollama_version": version,
        "hardware": f"{_cpu_name()}, {os.cpu_count()} threads, model on {placement}",
        "max_tool_calls": agent.max_tool_calls,
        "warmup_s": round(warmup_s, 1),
    })
    _atomic_write(META, json.dumps(meta, indent=2))
    print(f"Warm-up {warmup_s:.1f} s. Running {len(pending)} question(s).\n", flush=True)

    if not RESULTS.exists() or RESULTS.stat().st_size == 0:
        _durable_append(RESULTS, _csv_text([], header=True))

    for n, question in enumerate(pending, start=1):
        print(f"[{n}/{len(pending)}] {question['id']}: {question['question'][:90]}", flush=True)
        start = time.perf_counter()
        try:
            result = agent.ask(question["question"])
        except AgentError as exc:
            row = error_row(question, str(exc), time.perf_counter() - start)
        except Exception as exc:  # a bug must not cost the questions already answered
            row = error_row(question, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        else:
            broken = [c for c in result.tool_calls if is_infrastructure_error(c.error_code)]
            if broken:  # an API outage or a missing index says nothing about the model: retry later
                failures = "; ".join(f"{c.name}: {c.error_code}" for c in broken)
                row = error_row(question, f"tool infrastructure failure ({failures})", result.elapsed_s)
            else:
                try:
                    row = score(question, result)
                except Exception as exc:  # a scoring bug must not stop the run either
                    row = error_row(question, f"scoring failed: {type(exc).__name__}: {exc}", result.elapsed_s)
                else:
                    trace = {"id": question["id"], **result.to_dict()}
                    _durable_append(TRACES, json.dumps(trace, ensure_ascii=False, default=str) + "\n")
        _durable_append(RESULTS, _csv_text([row], header=False))

        if row["error"]:
            print(f"      ERROR after {row['time_s']} s: {row['error']}\n", flush=True)
        else:
            print(
                f"      tools={row['tools_called'] or 'none'} match={row['tool_match']} "
                f"cites={row['cites_source']} correct={row['answer_correct']} "
                f"stop={row['stop_reason']} {row['time_s']} s\n",
                flush=True,
            )

    meta["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    _atomic_write(META, json.dumps(meta, indent=2))
    _keep_traces({i for i, row in current_results().items() if finished(row)})
    print(write_summary(questions, meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
