"""Tool-calling agent: a local Ollama model chooses tools and answers only from their output.

Per question:

1. The conversation and ``TOOL_SCHEMAS`` go to the model.
2. Every tool call it makes is executed (at most ``max_tool_calls`` per question) and a
   compact digest of the result is fed back. Each source in a digest carries a short tag:
   ``[L1]`` for a literature passage, ``[ref11_LN-SDC]`` for a dataset paper, ``[MP1]`` for
   a Materials Project entry, ``[A1]`` for an Arrhenius fit.
3. When the model answers, the answer is checked against the tags it was actually given.

The checks are what make a 3B model usable for research answers:

* no tool called      -> nudge once, then refuse instead of answering from memory;
* nothing to cite     -> every tool came back empty, so the answer is the not-found sentence;
* no valid citation   -> ask once for a cited rewrite, then flag the answer as unverified;
* invented citations  -> replaced with "[unverified source]" and counted.

Valid tags are expanded into full citations (paper title and page, dataset ``paper_id``,
Materials Project id) and the cited sources are listed under the answer.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from . import config
from .numbers import ANY_DIGITS, normalize, number_tokens, numbers_in, quantities, reports, to_float
from .tools import TOOL_SCHEMAS, call_tool

NOT_FOUND = "The available tools do not contain this information."

SYSTEM_PROMPT = f"""You are a research assistant for solid oxide fuel cell (SOFC) materials. You can only answer from the results of these tools:
- search_literature: what the research papers say (mechanisms, explanations, methods).
- query_dataset: values measured in the papers (conductivity, activation energy, peak power, OCV) for doped ceria and carbonate composites.
- arrhenius: activation energy from conductivity values the user gives at several temperatures.
- materials_project: stability and band gap of a pure compound such as CeO2, Sm2O3 or Na2CO3.

Rules:
1. Always call a tool before answering. If the question has two parts, call a tool for each part.
2. Use only facts from the tool results. Never use your own knowledge and never invent numbers.
3. After each fact, put the source tag from the tool result in square brackets, for example [L1], [ref11_LN-SDC], [MP1] or [A1].
4. If the tool results do not answer the question, reply exactly: {NOT_FOUND}
5. Answer in at most 5 sentences."""

NUDGE = (
    "You have not called a tool. Call one of the tools now; do not answer from your own knowledge."
)
REPAIR = (
    "Your answer has no source tags. Rewrite it using only facts from the tool results and put the "
    "matching tag in square brackets after each fact, for example [L1] or [ref11_LN-SDC]. If the "
    f"tool results do not answer the question, reply exactly: {NOT_FOUND}"
)
LIMIT_REACHED = (
    "You have used all the tool calls allowed for this question. Write the final answer now, "
    "using only the tool results above and their source tags."
)
FORMAT_HINT = (
    "Your last message looks like a tool call, but it could not be read. Call the tool again "
    "with valid JSON arguments, or write the final answer from the tool results."
)

# Brackets and parentheses that may hold citations.
_BRACKETED = re.compile(r"\[([^\[\]\n]{1,300})\]")
_PAREN_TAGS = re.compile(
    r"\(\s*((?:L\d+|MP\d+|A\d+|mp-[a-z0-9]+)(?:\s*[,;]\s*(?:L\d+|MP\d+|A\d+|mp-[a-z0-9]+))*)\s*\)", re.I
)
# Bare tags in prose count only for passages and Materials Project entries: "A1" is too
# common in ordinary text ("the A1 site") to be read as a citation.
_BARE_TAG = re.compile(r"(?<![\w\[/-])((?:L|MP)\d+)(?![\w\]-])", re.I)
_TAGLIKE = re.compile(r"^(?:L\d+|MP\d+|A\d+|ref\d+[\w.+-]*|mp-[a-z0-9]+)$", re.I)
_TAG_TOKEN = re.compile(r"(?<![A-Za-z0-9_])(?:L\d+|MP\d+|A\d+|ref\d+[\w.+-]*|mp-[a-z0-9]+)(?![A-Za-z0-9_])", re.I)
_REF_ID = re.compile(r"^(ref\d+)(?:_[\w.+-]*)?$", re.I)
_TITLE_PAGE = re.compile(r"^(?P<title>.+?),\s*p(?:age)?\.?\s*(?P<page>\d+)$", re.I)
_TOOL_ALIASES = {
    "materials project": "materials_project",
    "materials_project": "materials_project",
    "arrhenius": "arrhenius",
    "arrhenius fit": "arrhenius",
}

# Phrases that state an absence of data. The subject must be the tools or their output
# ("the dataset does not contain", "no rows", "there are no Gd-doped rows"), so a real
# answer such as "there is no electronic short circuit" is not read as a decline.
_SUBJECT = (
    r"(?:tools?|dataset|data\s?set|papers?|passages?|results?|corpus|literature|"
    r"materials project|database|table|sources?|documents?|it|they)"
)
_DATA_NOUN = (
    r"(?:data|information|entr(?:y|ies)|rows?|values?|measurements?|results?|passages?|records?|"
    r"mention|match(?:es)?)"
)
_DECLINE = re.compile(
    rf"\b(?:{_SUBJECT}\s+(?:do|does|did)(?:\s+not|n't)\s+(?:contain|include|report|mention|give|list|"
    r"index|provide|cover|have|say|discuss)|"
    rf"not\s+(?:found|reported|available|mentioned|listed|covered|provided|given)\s+(?:in|by)\s+(?:the\s+)?{_SUBJECT}|"
    r"not\s+indexed|"
    rf"no\s+(?:matching\s+)?{_DATA_NOUN}\b|"
    rf"there\s+(?:is|are)\s+no\s+(?:\S+\s+){{0,3}}?{_DATA_NOUN}\b|"
    r"none\s+of\s+the\s+(?:\d+\s+)?(?:rows?|passages?|papers?|entries|results?|records?|compositions?)|"
    r"cannot\s+(?:answer|be\s+determined|be\s+calculated|be\s+found|determine|calculate|compute|find)|"
    r"can't\s+(?:answer|determine|calculate)|unable\s+to\s+(?:answer|find|determine|calculate)|"
    r"(?:could\s+not|couldn't)\s+find|not\s+enough\s+(?:data|information|points|values)|"
    r"insufficient\s+(?:data|information)|"
    r"at\s+least\s+(?:two|2)\s+(?:\S+\s+)?(?:points|temperatures|values|measurements|data)|"
    r"(?:two|2)\s+or\s+more\s+(?:\S+\s+)?(?:points|temperatures|values|measurements)|"
    r"can(?:'|no)t\s+be\s+(?:calculated|determined|computed)|"
    r"not\s+enough\s+(?:to\s+(?:calculate|determine|compute|fit)|measurements)|"
    r"not\s+possible\s+to\s+(?:determine|calculate))",
    re.I,
)
# A measured value: a number with a physical unit (temperatures excluded, because a
# decline may still say where the data it lacks would be, e.g. "at 600 C").
_STATED_VALUE = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<num>\d+(?:\.\d+)?(?:\s*x\s*10\^?-?\d+)?)\s*"
    r"(?:m?S\s*/\s*cm|m?S\s*cm|m?eV|mW|W\s*/\s*cm|kJ|mV|V\b|g\s*/\s*cm)",
)


class AgentError(RuntimeError):
    """The model backend could not be reached or rejected the request."""


# chat(messages, tools) -> {"content": str, "tool_calls": list, "usage": {...}}
ChatFn = Callable[[list, "list | None"], dict]


class OllamaChat:
    """Minimal client for Ollama's ``/api/chat`` endpoint with tool calling."""

    def __init__(
        self,
        model: str | None = None,
        url: str | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        seed: int | None = None,
        timeout_s: float | None = None,
        keep_alive: str | None = None,
    ) -> None:
        def pick(value: Any, key: str, default: Any) -> Any:
            return config.get(key, default) if value is None else value

        self.model = str(pick(model, "agent.model", "qwen2.5:3b-instruct"))
        self.url = str(pick(url, "agent.ollama_url", "http://127.0.0.1:11434")).rstrip("/")
        self.options = {
            "temperature": float(pick(temperature, "agent.temperature", 0.0)),
            "num_ctx": int(pick(num_ctx, "agent.num_ctx", 8192)),
            "num_predict": int(pick(num_predict, "agent.num_predict", 400)),
            "seed": int(pick(seed, "agent.seed", 42)),
        }
        self.timeout_s = float(pick(timeout_s, "agent.timeout_s", 600))
        self.keep_alive = str(pick(keep_alive, "agent.keep_alive", "30m"))

    def __call__(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        import requests  # deferred so importing the agent stays cheap

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self.options,
            "keep_alive": self.keep_alive,
        }
        if tools:
            payload["tools"] = tools
        try:
            response = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout_s)
        except requests.exceptions.ConnectionError as exc:
            raise AgentError(f"cannot reach Ollama at {self.url} - is `ollama serve` running?") from exc
        except requests.exceptions.Timeout as exc:
            raise AgentError(f"Ollama did not answer within {self.timeout_s:.0f} s") from exc
        except requests.exceptions.RequestException as exc:
            raise AgentError(f"request to Ollama failed: {exc}") from exc
        if response.status_code == 404:
            if "model" in response.text.lower():
                raise AgentError(f"model '{self.model}' is not installed - run `ollama pull {self.model}`")
            raise AgentError(f"no Ollama chat endpoint at {self.url}/api/chat - check agent.ollama_url")
        if response.status_code >= 400:
            raise AgentError(f"Ollama returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise AgentError(f"Ollama returned a non-JSON reply: {response.text[:200]!r}") from exc
        message = data.get("message") or {}
        return {
            "content": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
            "usage": {
                "prompt_tokens": int(data.get("prompt_eval_count") or 0),
                "completion_tokens": int(data.get("eval_count") or 0),
            },
        }


@dataclass
class Source:
    """One citable item handed to the model under a tag."""

    tag: str
    tool: str
    inline: str  # what the tag becomes inside the answer text
    detail: str  # one line in the Sources list
    title: str | None = None  # literature only, for title-and-page citations
    page: int | None = None
    aliases: tuple[str, ...] = ()  # other spellings that identify it, e.g. "mp-aaaabdws"


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict
    ok: bool
    summary: str
    elapsed_s: float
    tags: list[str] = field(default_factory=list)
    repeated: bool = False
    ignored_arguments: list[str] = field(default_factory=list)
    error_code: str | None = None


# Failures of the tools' own infrastructure, not of the model's choices. The evaluation
# retries a question hit by one of these instead of scoring it.
INFRASTRUCTURE_ERRORS = frozenset({  # not "tool_exception": a bug or bad input repeats on retry
    "network_error", "rate_limited", "unauthorized", "missing_api_key", "bad_json",
    "index_unavailable", "dataset_unavailable",
})


def is_infrastructure_error(code: str | None) -> bool:
    return bool(code) and (code in INFRASTRUCTURE_ERRORS or code.startswith("http_5"))


@dataclass
class Citations:
    text: str
    valid: list[Source]
    invalid: list[str]


@dataclass
class AgentResult:
    question: str
    answer: str  # what the user sees: body + Sources (or the not-found block)
    answer_body: str  # the answer text alone, citations expanded, no appended lists
    model_answer: str  # the model's raw final reply, tags unexpanded
    tool_calls: list[ToolCallRecord]
    cited_sources: list[Source]
    invalid_citations: list[str]
    ungrounded_numbers: list[str]
    cited: bool
    not_found: bool
    nudged: bool
    repaired: bool
    stop_reason: str
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    model: str

    @property
    def tools_used(self) -> list[str]:
        """Tool names in the order first called."""
        return list(dict.fromkeys(call.name for call in self.tool_calls))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tools_used"] = self.tools_used
        return data


class _Registry:
    """Every source tag handed to the model during one question."""

    def __init__(self) -> None:
        self._by_tag: dict[str, Source] = {}
        self._counters = {"L": 0, "MP": 0, "A": 0}
        self._passages: dict[tuple, str] = {}

    def __bool__(self) -> bool:
        return bool(self._by_tag)

    def new_tag(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix}{self._counters[prefix]}"

    def add(self, source: Source) -> None:
        self._by_tag.setdefault(source.tag.lower(), source)

    def get(self, tag: str) -> Source | None:
        return self._by_tag.get(tag.strip().lower())

    def sources(self) -> list[Source]:
        return list(self._by_tag.values())

    def passage_tag(self, key: tuple) -> str | None:
        return self._passages.get(key)

    def remember_passage(self, key: tuple, tag: str) -> None:
        self._passages[key] = tag


class _Grounding:
    """Numbers an Arrhenius fit may legitimately use, kept by role.

    Only the question and data that tools actually produced go in: never an error or
    refusal message, an echoed query string or the filters a model asked for, because
    those repeat the model's own (possibly invented) numbers back to it.
    """

    def __init__(self, question: str) -> None:
        # Numbers a person wrote (the question, a paper), sorted by the unit written after
        # them or after their list. A number with some other unit ("20 mol%", "5 h") or
        # glued to a name ("20SDC") is neither a temperature nor a conductivity.
        self.written_temperatures: list[float] = []
        self.written_kelvin: list[float] = []  # the subset written with K
        self.written_conductivities: list[float] = []  # may appear rescaled (mS/cm, S/m)
        self.written_bare: list[float] = []  # no unit: accepted only as written
        self._read(question)
        # Values tools produced: temperatures in C, conductivities already in S/cm.
        self.temperatures: list[float] = []
        self.conductivities: list[float] = []

    def _read(self, text: str) -> None:
        for value, kind, unit in quantities(text):
            if kind == "temperature":
                self.written_temperatures.append(value)
                if unit == "K":
                    self.written_kelvin.append(value)
            elif kind == "conductivity":
                self.written_conductivities.append(value)
            elif kind == "bare":
                self.written_bare.append(value)

    def add(self, name: str, result: dict) -> None:
        if not result.get("ok"):
            return
        if name == "query_dataset":
            for row in result.get("rows") or []:
                if row.get("conductivity_temp_C") is not None:
                    self.temperatures.append(float(row["conductivity_temp_C"]))
                if row.get("conductivity_S_cm") is not None:
                    self.conductivities.append(float(row["conductivity_S_cm"]))
        elif name == "search_literature":
            for passage in relevant_passages(result):
                self._read(passage.get("snippet", ""))
        elif name == "arrhenius":
            for point in result.get("points") or []:
                self.temperatures.append(point["temperature_C"])
                self.conductivities.append(point["conductivity_S_cm"])

    @staticmethod
    def _near(value: float, candidates: list[float]) -> bool:
        return any(abs(c - value) <= 1e-3 * max(abs(value), 1e-12) for c in candidates)

    def written_in_kelvin(self, temperatures: list[float]) -> bool:
        """Were these exact numbers written in kelvin, and none of them in Celsius?"""
        celsius = [t for t in self.written_temperatures if not self._near(t, self.written_kelvin)]
        return bool(temperatures) and all(
            self._near(t, self.written_kelvin) and not self._near(t, celsius) for t in temperatures
        )

    def invented(self, temperatures: list[float], conductivities: list[float]) -> list[float]:
        """Values with no source.

        Any temperature may switch between C and K. A conductivity a person wrote with a
        conductivity unit may also switch scale (mS/cm, S/m; the scale leaves Ea unchanged).
        Unit-less numbers and tool values are accepted only as they are: rescaling them
        would let "20 mol%" ground 0.02 and 0.2, or 0.12 S/cm ground 0.00012.
        """
        kelvin_shift = lambda xs: [c for x in xs for c in (x, x + 273.15, x - 273.15)]  # noqa: E731
        t_pool = kelvin_shift(self.written_temperatures + self.written_bare + self.temperatures)
        s_pool = [c for x in self.written_conductivities for c in (x, x * 1e-3, x * 1e3, x * 1e-2, x * 1e2)]
        s_pool += self.written_bare + self.conductivities
        return [t for t in temperatures if not self._near(t, t_pool)] + [
            s for s in conductivities if not self._near(s, s_pool)
        ]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


# --------------------------------------------------------------------------- digests

def relevant_passages(result: dict) -> list[dict]:
    """Passages good enough to show the model: none unless the best one reaches
    agent.min_top_passage_score, and then only those above agent.min_passage_score."""
    passages = result.get("passages") or []
    top = max((p.get("score", 0) for p in passages), default=0)
    if top < float(config.get("agent.min_top_passage_score", 0.50)):
        return []
    threshold = float(config.get("agent.min_passage_score", 0.40))
    return [p for p in passages if p.get("score", 0) >= threshold]


def _digest_literature(result: dict, reg: _Registry) -> tuple[str, list[str]]:
    kept = relevant_passages(result)
    dropped = len(result.get("passages", [])) - len(kept)
    if not kept:
        # The query is not repeated here: a model's query can carry invented numbers.
        return (
            "search_literature found no passage relevant to this question in the corpus. "
            "This result does not answer the question."
        ), []
    lines = [f'search_literature results for "{result.get("query")}". Cite a passage by its tag, e.g. [L1].']
    tags: list[str] = []
    for p in kept:
        key = (p.get("file_name"), p.get("page"), (p.get("snippet") or "")[:80])
        tag = reg.passage_tag(key)
        if tag is None:
            tag = reg.new_tag("L")
            reg.remember_passage(key, tag)
            doi = f", DOI {p['doi']}" if p.get("doi") else ""
            reg.add(Source(
                tag=tag,
                tool="search_literature",
                inline=f"{p['title']}, p.{p['page']}",
                detail=f"{p['title']}, p.{p['page']} ({p.get('file_name')}{doi})",
                title=p["title"],
                page=int(p["page"]) if p.get("page") is not None else None,
            ))
        tags.append(tag)
        lines.append(f'[{tag}] "{p["title"]}", page {p["page"]} (relevance {p["score"]:.2f})\n    {p.get("snippet", "")}')
    if dropped:
        lines.append(f"({dropped} low-relevance passage(s) omitted.)")
    return "\n".join(lines), tags


def _dataset_row_line(row: dict) -> str:
    parts = [f"[{row['paper_id']}] {row.get('composition_label') or 'composition not recorded'}"]
    if row.get("ceria_dopant"):
        fraction = f" (fraction {_fmt(row['dopant_fraction'])})" if row.get("dopant_fraction") is not None else ""
        parts.append(f"dopant {row['ceria_dopant']}{fraction}")
    if row.get("secondary_phase"):
        parts.append(f"secondary phase {row['secondary_phase']}")
    if row.get("conductivity_S_cm") is not None:
        at = f" at {_fmt(row['conductivity_temp_C'])} C" if row.get("conductivity_temp_C") is not None else ""
        kind = f" ({row['conductivity_type']})" if row.get("conductivity_type") else ""
        parts.append(f"conductivity {_fmt(row['conductivity_S_cm'])} S/cm{at}{kind}")
    if row.get("activation_energy_eV") is not None:
        parts.append(f"activation energy {_fmt(row['activation_energy_eV'])} eV")
    if row.get("pmax_mW_cm2") is not None:
        at = f" at {_fmt(row['pmax_temp_C'])} C" if row.get("pmax_temp_C") is not None else ""
        parts.append(f"peak power density {_fmt(row['pmax_mW_cm2'])} mW/cm2{at}")
    if row.get("ocv_V") is not None:
        parts.append(f"OCV {_fmt(row['ocv_V'])} V")
    if row.get("atmosphere"):
        parts.append(f"atmosphere {row['atmosphere']}")
    return "; ".join(parts)


_MEASURED_FIELDS = ("conductivity_S_cm", "activation_energy_eV", "pmax_mW_cm2", "ocv_V")


def _register_paper(reg: _Registry, paper_id: str, rows: list[dict]) -> None:
    """Every paper_id that appears anywhere in a digest must be citable."""
    if reg.get(paper_id) is not None:
        return
    citation = next((r.get("citation") for r in rows if r.get("paper_id") == paper_id and r.get("citation")), None)
    reg.add(Source(
        tag=paper_id,
        tool="query_dataset",
        inline=paper_id,
        detail=f"{paper_id}: {citation or 'citation not recorded'} (extracted dataset)",
    ))


def _digest_dataset(result: dict, reg: _Registry) -> tuple[str, list[str]]:
    rows = result.get("rows") or []
    if not rows:
        return f"query_dataset: {result.get('summary')} This result does not answer the question.", []
    max_rows = int(config.get("agent.max_rows_in_prompt", 20))
    stats = result.get("statistics") or {}
    filters = result.get("filters_applied") or "no filters"
    # Rows with a measured value go first: the dataset lists samples in paper order and
    # several have no measurements, which would otherwise push real data past the cap.
    measured = [r for r in rows if any(r.get(f) is not None for f in _MEASURED_FIELDS)]
    unmeasured = [r for r in rows if r not in measured]
    shown = measured[:max_rows]

    # The header stays neutral: when it named the best row, a 3B model attributed that
    # row's value to whichever composition the question asked about.
    lines = [
        f"query_dataset: {stats.get('n_matches', len(rows))} of {result.get('dataset_rows_total')} "
        f"rows match {filters}. Each line is one measured composition; cite it by its paper_id "
        f"in square brackets, e.g. [{(shown or rows)[0]['paper_id']}].",
    ]
    tags: list[str] = []
    for row in shown:
        _register_paper(reg, row["paper_id"], rows)
        if row["paper_id"] not in tags:
            tags.append(row["paper_id"])
        lines.append(_dataset_row_line(row))
    if unmeasured:
        labels = "; ".join(f"{r.get('composition_label')} [{r['paper_id']}]" for r in unmeasured[:5])
        for row in unmeasured[:5]:
            _register_paper(reg, row["paper_id"], rows)
        lines.append(f"({len(unmeasured)} matching row(s) report no measured values: {labels}.)")
    not_shown = int(stats.get("n_matches", len(rows))) - len(shown) - len(unmeasured)
    if not_shown > 0:
        lines.append(f"({not_shown} more matching row(s) not shown; add filters to narrow the query.)")
    best = stats.get("best_conductivity")
    if best and int(stats.get("n_with_conductivity", 0)) > 1:
        _register_paper(reg, best["paper_id"], rows)
        lines.append(
            f"Statistic over all matching rows - highest conductivity: {_fmt(best['conductivity_S_cm'])} S/cm "
            f"at {_fmt(best['conductivity_temp_C'])} C, composition {best['composition_label']} "
            f"[{best['paper_id']}]."
        )
    best_pmax = stats.get("best_pmax")
    if best_pmax and int(stats.get("n_with_pmax", 0)) > 1:
        _register_paper(reg, best_pmax["paper_id"], rows)
        lines.append(
            f"Statistic over all matching rows - highest peak power density: "
            f"{_fmt(best_pmax['pmax_mW_cm2'])} mW/cm2 at {_fmt(best_pmax['pmax_temp_C'])} C, "
            f"composition {best_pmax['composition_label']} [{best_pmax['paper_id']}]."
        )
    return "\n".join(lines), tags


def _digest_materials(result: dict, reg: _Registry) -> tuple[str, list[str]]:
    if not result.get("found"):
        return f"materials_project: {result.get('summary')} This result does not answer the question.", []
    best = result["most_stable"]
    tag = reg.new_tag("MP")
    reg.add(Source(
        tag=tag,
        tool="materials_project",
        inline=f"Materials Project {best['material_id']}",
        detail=(
            f"Materials Project {best['material_id']}: {best['formula_pretty']} "
            f"{best.get('spacegroup_symbol') or ''} ({best.get('mp_url')})"
        ),
        aliases=(str(best["material_id"]).lower(),),
    ))
    polymorphs = ", ".join(
        f"{p.get('spacegroup_symbol')} (+{_fmt(p.get('energy_above_hull_eV_per_atom'))} eV/atom)"
        for p in result.get("polymorphs", [])[:4]
    )
    text = (
        f"materials_project result. Cite it as [{tag}].\n"
        f"[{tag}] {best['formula_pretty']}, Materials Project {best['material_id']}, space group "
        f"{best.get('spacegroup_symbol')} ({best.get('crystal_system')}): {result.get('stability')}; "
        f"{result.get('band_gap_description')}; formation energy "
        f"{_fmt(best.get('formation_energy_eV_per_atom'))} eV/atom; energy above hull "
        f"{_fmt(best.get('energy_above_hull_eV_per_atom'))} eV/atom."
    )
    if polymorphs:
        text += f" Less stable polymorphs: {polymorphs}."
    return text, [tag]


def _digest_arrhenius(result: dict, reg: _Registry) -> tuple[str, list[str]]:
    tag = reg.new_tag("A")
    inline = "Arrhenius fit" if tag == "A1" else f"Arrhenius fit {tag[1:]}"
    reg.add(Source(
        tag=tag,
        tool="arrhenius",
        inline=inline,
        detail=f"{inline}: computed from the supplied conductivity data - {result.get('summary')}",
    ))
    text = f"arrhenius result. Cite it as [{tag}].\n[{tag}] {result.get('summary')}"
    if result.get("notes"):
        text += " Notes: " + " ".join(result["notes"])
    return text, [tag]


_DIGESTERS = {
    "search_literature": _digest_literature,
    "query_dataset": _digest_dataset,
    "materials_project": _digest_materials,
    "arrhenius": _digest_arrhenius,
}


def digest(name: str, result: dict, reg: _Registry) -> tuple[str, list[str]]:
    """Compact, tagged rendering of a tool result for the model to read."""
    ignored = result.get("ignored_arguments")
    note = f" (ignored unknown arguments: {', '.join(ignored)})" if ignored else ""
    if not result.get("ok"):
        return (
            f"{name} failed: {result.get('error', 'unknown error')}{note}. "
            "Correct the arguments or use a different tool."
        ), []
    text, tags = _DIGESTERS[name](result, reg)
    return text + note, tags


# ------------------------------------------------------------------- tool-call parsing

def _coerce_arguments(arguments: Any) -> dict:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def _balanced_end(text: str, start: int) -> int | None:
    """Index just past the brace that closes the one at ``start`` (quote-aware)."""
    depth, quote, escaped = 0, "", False
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _call_in(obj: Any, names: set[str]) -> tuple[str, dict] | None:
    """{"name": ..., "arguments": ...} or the nested {"function": {"name": ..., ...}} form."""
    if not isinstance(obj, dict):
        return None
    body = obj.get("function") if isinstance(obj.get("function"), dict) else obj
    name = body.get("name")
    if not isinstance(name, str) and isinstance(obj.get("function"), str):
        name, body = obj["function"], obj
    if isinstance(name, str) and name in names:
        return name, _coerce_arguments(body.get("arguments", body.get("parameters", {})))
    return None


def _text_tool_calls(text: str, names: set[str]) -> list[tuple[str, dict, int, int]]:
    """(name, arguments, start, end) for each tool call written in reply text: JSON, or a
    Python-style dict with single quotes, which small models also produce."""
    import ast

    found: list[tuple[str, dict, int, int]] = []
    decoder = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except ValueError:
            end = _balanced_end(text, i)
            try:
                obj = ast.literal_eval(text[i:end]) if end else None
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                obj = None
            if obj is None:
                i = text.find("{", i + 1)
                continue
        call = _call_in(obj, names)
        if call:
            found.append((call[0], call[1], i, end))
        i = text.find("{", end)
    return found


def looks_like_tool_call(text: str, names: set[str]) -> bool:
    """A reply that tries to call a tool, even if it could not be parsed."""
    if "<tool_call>" in text or "</tool_call>" in text:
        return True
    return any(re.search(rf"[\"']name[\"']\s*:\s*[\"']{re.escape(n)}[\"']", text) for n in names)


def _tool_calls_in_text(text: str, names: set[str]) -> list[tuple[str, dict]]:
    """A small model sometimes writes the call as JSON in its reply instead of tool_calls."""
    return [(name, arguments) for name, arguments, _, _ in _text_tool_calls(text, names)]


def strip_tool_calls(text: str, names: set[str]) -> str:
    """The reply with any text tool calls (and their <tool_call> wrappers) removed."""
    for _, _, start, end in reversed(_text_tool_calls(text, names)):
        text = text[:start] + text[end:]
    return re.sub(r"</?tool_call>", " ", text).strip()


def _coerce_to_schema(arguments: dict, types: dict[str, str | None]) -> tuple[dict, str | None]:
    """Fit a small model's argument shapes to the schema types.

    ['CeO2'] -> 'CeO2'; ['Sm', 'Ca'] for a string -> 'Sm or Ca'; {'type': 'string',
    'value': 'CeO2'} (a schema echoed back) -> 'CeO2'; 600 for a string -> '600'.
    Returns (arguments, error message or None).
    """
    out = dict(arguments)
    for key, value in arguments.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            del out[key]  # an explicit null or "" for an optional argument means "not given"
            continue
        expected = types.get(key)
        if isinstance(value, dict) and "value" in value and expected != "object":
            value = value["value"]
        if expected in ("string", "number", "integer") and isinstance(value, list) and len(value) == 1:
            value = value[0]
        if expected == "string" and value is not None and not isinstance(value, str):
            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                value = " or ".join(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                value = f"{value:g}" if isinstance(value, float) else str(value)
            else:
                return out, f"{key} must be a string, got {type(value).__name__}"
        out[key] = value
    return out, None


def extract_tool_calls(reply: dict, names: set[str]) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []
    for call in reply.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") or call.get("name")
        if isinstance(name, str) and name:
            calls.append((name, _coerce_arguments(function.get("arguments", call.get("arguments", {})))))
    if calls:
        return calls
    return _tool_calls_in_text(reply.get("content") or "", names)


# ------------------------------------------------------------------ citation checking

def _lookup(token: str, reg: _Registry) -> Source | None:
    text = token.strip().strip("\"'`* ").rstrip(".,;:")
    if not text:
        return None
    source = reg.get(text)
    if source:
        return source

    tool = _TOOL_ALIASES.get(text.lower())
    if tool:  # "[Materials Project]" is traceable only when there is exactly one such source
        matches = [s for s in reg.sources() if s.tool == tool]
        return matches[0] if len(matches) == 1 else None

    low = text.lower()
    for s in reg.sources():  # "[Materials Project mp-aaaabdws]", "[mp-aaaabdws]", "[Arrhenius fit 2]"
        if low == s.inline.lower() or low in s.aliases:
            return s

    short_id = _REF_ID.match(text)
    if short_id:  # "[ref11]" for "[ref11_LN-SDC]" - only when the whole token is a paper id
        prefix = short_id.group(1).lower()
        matches = [s for s in reg.sources() if s.tool == "query_dataset" and s.tag.lower().split("_")[0] == prefix]
        return matches[0] if len(matches) == 1 else None

    title_page = _TITLE_PAGE.match(text)
    if title_page:  # "[Improved ceria-carbonate composite electrolytes, p.1]"
        want = _norm(title_page.group("title"))
        page = int(title_page.group("page"))
        if len(want) >= 12:
            for s in reg.sources():
                if s.tool == "search_literature" and s.page == page:
                    have = _norm(s.title or "")
                    if have.startswith(want) or want.startswith(have):
                        return s
    return None


def resolve_citations(text: str, reg: _Registry) -> Citations:
    """Expand valid source tags into full citations; flag the invented ones."""
    valid: list[Source] = []
    invalid: list[str] = []

    def keep(source: Source) -> None:
        if source not in valid:
            valid.append(source)

    def expand(inner: str, original: str) -> str:
        whole = _lookup(inner, reg)
        if whole:
            keep(whole)
            return f"[{whole.inline}]"
        found: list[Source] = []
        bad: list[str] = []
        prose: list[str] = []
        # ';' first so "Title, p.4; MP1" keeps its title citation whole.
        for chunk in inner.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            source = _lookup(chunk, reg)
            if source:
                found.append(source)
                continue
            tokens = list(_TAG_TOKEN.finditer(chunk))
            if not tokens:
                parts = [p.strip() for p in chunk.split(",") if p.strip()]
                resolved = [_lookup(p, reg) for p in parts]
                if any(resolved):  # "Title A, p.1" inside a longer chunk
                    found += [s for s in resolved if s]
                elif _TITLE_PAGE.match(chunk):
                    bad.append(chunk)  # a title and page that no tool returned
                else:
                    prose.append(chunk)
                continue
            # "[L1 and L2]", "[Source: L1]", "[L1/L2]", "[MP1 (CeO2)]", "[ref11_LN-SDC and ref99_X]"
            for token in tokens:
                source = _lookup(token.group(), reg)
                if source:
                    found.append(source)
                else:
                    bad.append(token.group())
            rest = _TAG_TOKEN.sub(" ", chunk).strip(" :/&,.()")
            also = _lookup(rest, reg) if rest else None  # "[L1: Title, p.1]"
            if also:
                found.append(also)
                continue
            rest = re.sub(r"\b(?:sources?|see|cf|and|tags?|from)\b|[:/&,()]", " ", rest, flags=re.I)
            rest = re.sub(r"\s+", " ", rest).strip(" .")
            if re.search(r"[A-Za-z]{3,}|\d", rest):  # "(CeO2)", "at 600 C": keep; "and", ":" drop
                prose.append(rest)
        if not found and not bad:
            return original  # ordinary bracketed text, not a citation
        for source in found:
            keep(source)
        invalid.extend(bad)
        cited = "[" + "; ".join(dict.fromkeys(s.inline for s in found)) + "]" if found else "[unverified source]"
        return cited + (f" ({', '.join(prose)})" if prose else "")  # keep words like "at 600 C"

    body = _BRACKETED.sub(lambda m: expand(m.group(1), m.group(0)), text)
    body = _PAREN_TAGS.sub(lambda m: expand(m.group(1), m.group(0)), body)

    def bare(match: re.Match) -> str:
        source = reg.get(match.group(1))
        if source is None:
            return match.group(0)
        keep(source)
        return f"[{source.inline}]"

    body = _BARE_TAG.sub(bare, body)
    for source in reg.sources():  # a paper_id or mp-id named without brackets still traces
        names = [source.tag] if source.tool == "query_dataset" else list(source.aliases)
        if source.tool in ("query_dataset", "materials_project") and any(
            re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", body, re.I) for n in names
        ):
            keep(source)
    return Citations(text=body, valid=valid, invalid=invalid)


def states_value(text: str, known: str = "") -> bool:
    """Does the text state a measured value (a number with a unit) outside citations?
    Values that also appear in ``known`` (the question) are only restated, not claimed."""
    known_values = numbers_in(known)
    for match in _STATED_VALUE.finditer(normalize(_BRACKETED.sub(" ", text or ""))):
        try:
            value = to_float(match.group("num"))
        except ValueError:
            continue
        if not any(abs(value - k) <= 1e-9 * max(abs(k), 1.0) for k in known_values):
            return True
    return False


_LEADING_NOT_FOUND = re.compile(
    r"^\s*(?:\[[^\]]*\]\s*)*" + re.escape(NOT_FOUND.rstrip(".")) + r"[^.\n]*\.?\s*(?:\[[^\]]*\]\s*)*\.?\s*",
    re.I,
)


def _is_not_found_sentence(text: str) -> bool:
    body = re.sub(r"\s+", " ", _BRACKETED.sub(" ", text or "")).strip().lower()
    return body.startswith(NOT_FOUND.lower().rstrip("."))


def says_not_found(text: str, max_words: int | None = None, known: str = "") -> bool:
    """Is this reply a statement that the answer is not available?

    True for the standard not-found sentence, or for a decline phrase ("the dataset does
    not contain...", "there are no rows...") when the reply claims no measured value of
    its own, so "There is no GDC entry, but GDC reaches 0.31 S/cm" is not a decline while
    "Ea cannot be determined from one value, 0.05 S/cm at 600 C" (restating the question,
    passed as ``known``) is. ``max_words`` limits the phrase test to short replies.
    """
    if _is_not_found_sentence(text):
        return True
    body = _BRACKETED.sub(" ", text or "")
    if max_words is not None and len(body.split()) > max_words:
        return False
    return _DECLINE.search(body) is not None and not states_value(body, known)


def ungrounded_numbers(answer: str, evidence: str) -> list[str]:
    """Numbers in the answer that are not a rounding of any number in the evidence.

    A heuristic hallucination check: 0.72 is grounded by 0.717, 69 by 69.18, 119 meV by
    0.119 eV, 873 K by 600 C and 0.013 by 1.3 x 10^-2. Numbers inside citation brackets
    and inside formulas (CeO2) are ignored.
    """
    values = {abs(v) for v in numbers_in(evidence)} | {float(n) for n in ANY_DIGITS.findall(evidence)}
    pool = set()
    for v in values:
        pool.update((v, v * 1000, v / 1000, v + 273.15, abs(v - 273.15)))
    missing: list[str] = []
    for token in number_tokens(_BRACKETED.sub(" ", answer or "")):
        token = token.lstrip("+-")
        if re.search(r"[xX^eE]", token):  # scientific notation: compare by value, 0.5 %
            try:
                value = abs(to_float(token))
            except ValueError:
                continue
            grounded = any(abs(value - c) <= 5e-3 * max(c, 1e-30) for c in pool)
        else:
            grounded = any(reports(token, candidate, allow_truncation=True) for candidate in pool)
        if not grounded and token not in missing:
            missing.append(token)
    return missing


def _evidence_text(name: str, result: dict) -> str:
    """The numbers a successful tool call actually produced, for the hallucination check.

    Built from the result data, not from the digest: digest headers repeat the model's
    own query and filters, and a number the model chose is not evidence for anything.
    """
    if name == "search_literature":
        return "\n".join(f"{p.get('title', '')} {p.get('snippet', '')}" for p in relevant_passages(result))
    if name == "query_dataset":
        rows = [" ".join(str(v) for k, v in row.items() if v is not None and k != "citation")
                for row in result.get("rows") or []]
        return "\n".join(rows)
    if name == "materials_project":
        return json.dumps({k: result.get(k) for k in ("most_stable", "polymorphs", "stability",
                                                       "band_gap_description")}, default=str)
    if name == "arrhenius":
        return json.dumps({k: result.get(k) for k in ("summary", "points", "activation_energy_eV",
                                                       "activation_energy_kJ_per_mol", "r_squared",
                                                       "prefactor_A")}, default=str)
    return str(result.get("summary", ""))


def _read_written_kelvin(arguments: dict, grounding: _Grounding) -> tuple[dict, str | None]:
    """A question gives '773 K, 823 K, 873 K' and the model passes [773, 823, 873] without
    temperature_unit: fitted as Celsius that is a cited, wrong Ea (1.86 instead of 1.06 eV).
    When every temperature was written in kelvin, and none of them in Celsius, fit in kelvin."""
    from .tools.arrhenius import _UNIT_ALIASES, _parse_items

    if _UNIT_ALIASES.get(str(arguments.get("temperature_unit", "C")).strip().lower()) != "C":
        return arguments, None
    try:
        items = _parse_items(arguments.get("temperatures"), "temperatures", "temperature")
    except ValueError:
        return arguments, None
    if any(unit is not None for _, unit in items):
        return arguments, None  # the items say which unit they are in
    if not grounding.written_in_kelvin([value for value, _ in items]):
        return arguments, None
    return {**arguments, "temperature_unit": "K"}, (
        "The temperatures were written in kelvin in the question, so they were fitted as kelvin."
    )


def _check_arrhenius_inputs(arguments: dict, grounding: _Grounding) -> str | None:
    """Return an error message if the fit would run on invented data, else None."""
    from .tools.arrhenius import _UNIT_ALIASES, parse_conductivities, parse_temperatures

    unit = _UNIT_ALIASES.get(str(arguments.get("temperature_unit", "C")).strip().lower(), "C")
    try:
        temperatures, _ = parse_temperatures(arguments.get("temperatures"), unit)
        conductivities = parse_conductivities(arguments.get("conductivities"))
    except ValueError:
        return None  # malformed input: let the tool itself explain what is wrong
    invented = grounding.invented(temperatures, conductivities)
    if not invented:
        return None
    # The rejected values are deliberately not repeated here: this text goes back to the
    # model, and echoing invented numbers would only make them look sourced next time.
    return (
        f"refused: {len(invented)} of the values given do not appear in the question or in "
        "earlier tool results. Only fit conductivity data that the user supplied; never invent "
        "measurements. If the question gives no conductivity series, do not call arrhenius."
    )


# ------------------------------------------------------------------------- the agent

class Agent:
    """Answer SOFC questions with a local tool-calling model.

    Args:
        chat: chat function; defaults to :class:`OllamaChat` built from config.yaml.
        max_tool_calls: tool executions allowed per question (default from config, 5).
        dispatch: tool dispatcher, ``call_tool`` by default; injectable for tests.
    """

    def __init__(
        self,
        chat: ChatFn | None = None,
        max_tool_calls: int | None = None,
        dispatch: Callable[[str, dict], dict] = call_tool,
        tool_schemas: list[dict] | None = None,
    ) -> None:
        self.chat = chat if chat is not None else OllamaChat()
        self.max_tool_calls = int(config.get("agent.max_tool_calls", 5) if max_tool_calls is None else max_tool_calls)
        self.dispatch = dispatch
        self.tool_schemas = TOOL_SCHEMAS if tool_schemas is None else tool_schemas
        self.tool_names = {entry["function"]["name"] for entry in self.tool_schemas}
        # Only schema-declared arguments reach a tool: parameters kept for Python callers
        # (limit, full_rows, ...) are not the model's to set.
        self.allowed_arguments = {
            entry["function"]["name"]: set(entry["function"]["parameters"].get("properties", {}))
            for entry in self.tool_schemas
        }
        self.parameter_types = {
            entry["function"]["name"]: {
                key: spec.get("type") for key, spec in entry["function"]["parameters"].get("properties", {}).items()
            }
            for entry in self.tool_schemas
        }
        self.model = getattr(self.chat, "model", "custom")

    def _run_tool(
        self, name: str, arguments: dict, reg: _Registry, cache: dict, grounding: _Grounding
    ) -> tuple[ToolCallRecord, str, dict | None]:
        """Run one call. Returns (record, digest for the model, raw result or None if cached)."""
        allowed = self.allowed_arguments.get(name)
        dropped: list[str] = []
        if allowed is not None:
            dropped = sorted(set(arguments) - allowed)
            arguments = {k: v for k, v in arguments.items() if k in allowed}
        arguments, type_error = _coerce_to_schema(arguments, self.parameter_types.get(name, {}))
        kelvin_note = None
        if name == "arrhenius" and not type_error:
            arguments, kelvin_note = _read_written_kelvin(arguments, grounding)
        key = name + json.dumps(arguments, sort_keys=True, default=str)
        if key in cache:
            text, tags, ok, summary, code = cache[key]
            record = ToolCallRecord(
                name, arguments, ok, summary, 0.0, list(tags), repeated=True,
                ignored_arguments=dropped, error_code=code,
            )
            return record, "You already made this exact call. Same result:\n" + text, None

        start = time.perf_counter()
        refusal = _check_arrhenius_inputs(arguments, grounding) if name == "arrhenius" else None
        if name not in self.tool_names:
            result = {
                "ok": False, "tool": name,
                "error": f"unknown tool {name!r}; available tools: {sorted(self.tool_names)}",
            }
        elif type_error:
            result = {"ok": False, "tool": name, "error": type_error}
        elif refusal:
            result = {"ok": False, "tool": name, "error": refusal}
        else:
            result = self.dispatch(name, arguments)
        elapsed = time.perf_counter() - start
        if dropped:
            result = dict(result)
            result["ignored_arguments"] = sorted(set(result.get("ignored_arguments") or []) | set(dropped))
        if kelvin_note and result.get("ok"):
            result = dict(result)
            result["notes"] = list(result.get("notes") or []) + [kelvin_note]
        if name in _DIGESTERS or not result.get("ok"):
            text, tags = digest(name, result, reg)
        else:  # a tool without a digester: show its summary, nothing citable
            text, tags = f"{name}: {result.get('summary', '')}", []
        summary = result.get("summary") if result.get("ok") else f"error: {result.get('error')}"
        code = None if result.get("ok") else result.get("error_code")
        cache[key] = (text, tags, bool(result.get("ok")), str(summary), code)
        record = ToolCallRecord(
            name, arguments, bool(result.get("ok")), str(summary), elapsed, tags,
            ignored_arguments=dropped, error_code=code,
        )
        return record, text, result

    def ask(self, question: str) -> AgentResult:
        start = time.perf_counter()
        question = (question or "").strip()
        reg = _Registry()
        grounding = _Grounding(question)
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        calls: list[ToolCallRecord] = []
        ok_evidence: list[str] = []  # data successful tools produced, for the number check
        cache: dict[str, tuple] = {}
        nudged = repaired = limit_sent = format_warned = False
        llm_calls = prompt_tokens = completion_tokens = 0
        # The reply being judged. It is cleared whenever a reply is rejected (nudge, repair)
        # or tools run, so a stale answer can never be what the loop finally returns.
        candidate = ""
        stop_reason = ""
        max_llm_calls = self.max_tool_calls + 5  # tool rounds + nudge + format hint + repair + limit

        while True:
            if llm_calls >= max_llm_calls:
                stop_reason = "llm_call_limit"
                break
            tools_allowed = len(calls) < self.max_tool_calls
            reply = self.chat(messages, self.tool_schemas if tools_allowed else None)
            llm_calls += 1
            usage = reply.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            content = (reply.get("content") or "").strip()
            requested = extract_tool_calls(reply, self.tool_names)
            candidate = ""

            if requested and tools_allowed:
                assistant: dict[str, Any] = {"role": "assistant", "content": content}
                if reply.get("tool_calls"):
                    assistant["tool_calls"] = reply["tool_calls"]
                messages.append(assistant)
                for name, arguments in requested:
                    if len(calls) >= self.max_tool_calls:
                        messages.append({
                            "role": "tool", "tool_name": name,
                            "content": "Not executed: the tool call limit for this question is reached.",
                        })
                        continue
                    record, text, result = self._run_tool(name, arguments, reg, cache, grounding)
                    calls.append(record)
                    if result is not None:
                        grounding.add(name, result)
                        if record.ok:
                            ok_evidence.append(_evidence_text(name, result))
                    messages.append({"role": "tool", "tool_name": name, "content": text})
                if len(calls) >= self.max_tool_calls and not limit_sent:
                    limit_sent = True
                    messages.append({"role": "user", "content": LIMIT_REACHED})
                continue

            if requested:  # over the limit and still asking for tools
                content = strip_tool_calls(content, self.tool_names)
                if not content:
                    if limit_sent:
                        stop_reason = "tool_limit"
                        break
                    limit_sent = True
                    messages.append({"role": "assistant", "content": reply.get("content") or ""})
                    messages.append({"role": "user", "content": LIMIT_REACHED})
                    continue
                # otherwise the text that came with the call is judged as the answer

            if looks_like_tool_call(content, self.tool_names):  # a call we could not read
                if not format_warned:
                    format_warned = True
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": FORMAT_HINT})
                    continue
                candidate = content
                stop_reason = "malformed_tool_call"
                break

            candidate = content
            if not calls:
                if not nudged:
                    nudged = True
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": NUDGE})
                    candidate = ""
                    continue
                stop_reason = "no_tool_called"
                break
            if not reg:
                stop_reason = "nothing_to_cite"
                break
            if _is_not_found_sentence(content):
                # "<not-found sentence> [L1]" is a decline; "<not-found sentence> for part B.
                # Part A is 0.052 S/cm [ref11_LN-SDC]." is a partial, cited answer.
                rest = _LEADING_NOT_FOUND.sub("", content, count=1).strip()
                if not rest or says_not_found(rest, known=question) or not resolve_citations(rest, reg).valid:
                    stop_reason = "model_not_found"
                    break
            if content and resolve_citations(content, reg).valid:
                stop_reason = "answered"
                break
            if content and says_not_found(content, max_words=30, known=question):
                stop_reason = "model_not_found"
                break
            if not repaired:
                repaired = True
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": REPAIR})
                candidate = ""
                continue
            stop_reason = "uncited" if content else "empty_answer"
            break

        model_answer = candidate
        citations = resolve_citations(model_answer, reg) if calls else Citations(model_answer, [], [])
        not_found = stop_reason not in {"answered", "uncited"}
        evidence = question + "\n" + "\n".join(ok_evidence)

        if not_found:
            answer = body = NOT_FOUND
            if calls:
                answer += "\n\nWhat the tools returned:\n" + "\n".join(
                    f"- {call.name}: {call.summary}" for call in calls
                )
            grounded_check: list[str] = []
        else:
            answer = body = citations.text
            if citations.valid:
                answer += "\n\nSources:\n" + "\n".join(f"- {s.detail}" for s in citations.valid)
            else:
                answer += (
                    "\n\n(Unverified: the model did not cite any tool result for this answer.)\n"
                    "Tool results consulted:\n" + "\n".join(f"- {call.name}: {call.summary}" for call in calls)
                )
            grounded_check = ungrounded_numbers(citations.text, evidence)

        return AgentResult(
            question=question,
            answer=answer,
            answer_body=body,
            model_answer=model_answer,
            tool_calls=calls,
            cited_sources=citations.valid if not not_found else [],
            invalid_citations=citations.invalid,
            ungrounded_numbers=grounded_check,
            cited=bool(citations.valid) and not not_found,
            not_found=not_found,
            nudged=nudged,
            repaired=repaired,
            stop_reason=stop_reason,
            llm_calls=llm_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_s=time.perf_counter() - start,
            model=self.model,
        )
