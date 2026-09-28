"""Command line entry point: ``python -m sofc_agent "your question"``."""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from .agent import Agent, AgentError, OllamaChat


def _format_arguments(arguments: dict) -> str:
    parts = []
    for key, value in arguments.items():
        text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else repr(value)
        parts.append(f"{key}={text if len(text) <= 60 else text[:57] + '...'}")
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="python -m sofc_agent",
        description="Ask the local SOFC research agent a question. It picks tools, answers "
                    "only from their output, and cites what it used.",
    )
    parser.add_argument("question", nargs="+", help="the question (quotes optional)")
    parser.add_argument("--model", default=None,
                        help=f"Ollama model tag (default {config.get('agent.model')})")
    parser.add_argument("--max-tool-calls", type=int, default=None,
                        help=f"tool calls allowed per question (default {config.get('agent.max_tool_calls')})")
    parser.add_argument("--trace", action="store_true", help="also print the raw model answer and stop reason")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip()
    agent = Agent(chat=OllamaChat(model=args.model), max_tool_calls=args.max_tool_calls)
    try:
        result = agent.ask(question)
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(result.answer)
    print()
    if result.tool_calls:
        print("Tools called:")
        for call in result.tool_calls:
            flag = " (repeat)" if call.repeated else ""
            print(f"  {call.name}({_format_arguments(call.arguments)}){flag} -> {call.summary}")
    else:
        print("Tools called: none")
    if result.invalid_citations:
        print(f"Unverifiable citations removed: {', '.join(result.invalid_citations)}")
    if result.ungrounded_numbers:
        print(f"Numbers not found in any tool output: {', '.join(result.ungrounded_numbers)}")
    print(
        f"[{result.model} | {len(result.tool_calls)} tool call(s), {result.llm_calls} model call(s) | "
        f"{result.elapsed_s:.1f} s]"
    )
    if args.trace:
        print("\n--- trace ---")
        print(f"stop reason : {result.stop_reason}")
        print(f"nudged      : {result.nudged}   repaired: {result.repaired}")
        print(f"tokens      : {result.prompt_tokens} prompt, {result.completion_tokens} generated")
        print(f"model answer:\n{result.model_answer}")
    return 0
