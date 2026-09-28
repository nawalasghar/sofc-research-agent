"""Tool: query_dataset - structured questions over the extracted SOFC dataset.

The dataset is one row per reported composition/measurement, hand-extracted from the
papers, with ``paper_id`` linking every row back to its source. The source columns are
free text, and a small model phrases filters loosely, so each filter is parsed rather
than substring-matched:

* dopant: an element symbol or name ("Sm", "samarium", "SDC"), several of them ("Sm or
  Ca" means either, "Sm+Ca" means co-doped), a composition name as it appears in the
  dataset ("LN-SDC", "NK-CDC", "BSDC"), or a formula ("Ce0.8Sm0.2O1.9" -> Sm);
* carbonate: presence ("any", "no carbonate"), a carbonate system compared by its cations
  ("Na2CO3", "(Li/Na)2CO3", "Na2CO3-K2CO3", "sodium carbonate"), or "single"/"binary"/
  "ternary";
* temperature_C: one value in Celsius (matched within a tolerance), a value in kelvin
  ("873 K"), or a range ("500-600", "above 550").

Only the standard library is used, so this tool imports instantly.
"""
from __future__ import annotations

import csv
import math
import re
from functools import lru_cache
from typing import Any

from .. import config
from ..numbers import normalize, quantities

# Columns returned by default: enough to answer a question and cite it, without the
# long free-text fields (notes, synthesis_route) that would swamp a small model.
DEFAULT_FIELDS = (
    "paper_id",
    "citation",
    "composition_label",
    "ceria_dopant",
    "dopant_fraction",
    "secondary_phase",
    "secondary_phase_wt_pct",
    "conductivity_S_cm",
    "conductivity_temp_C",
    "conductivity_type",
    "activation_energy_eV",
    "pmax_mW_cm2",
    "pmax_temp_C",
    "ocv_V",
    "sintering_temp_C",
    "atmosphere",
    "fuel",
    "extraction_confidence",
)

NUMERIC_FIELDS = frozenset({
    "dopant_fraction",
    "secondary_phase_wt_pct",
    "sintering_temp_C",
    "pellet_thickness_um",
    "relative_density_pct",
    "conductivity_S_cm",
    "conductivity_temp_C",
    "activation_energy_eV",
    "pmax_mW_cm2",
    "pmax_temp_C",
    "ocv_V",
    "durability_h",
})

TEMPERATURE_FIELDS = ("conductivity_temp_C", "pmax_temp_C", "sintering_temp_C")

_ELEMENTS = frozenset(
    "h he li be b c n o f ne na mg al si p s cl ar k ca sc ti v cr mn fe co ni cu zn ga ge as "
    "se br kr rb sr y zr nb mo tc ru rh pd ag cd in sn sb te i xe cs ba la ce pr nd pm sm eu "
    "gd tb dy ho er tm yb lu hf ta w re os ir pt au hg tl pb bi po at rn fr ra ac th pa u np "
    "pu am cm bk cf es fm md no lr".split()
)

# ---------------------------------------------------------------------- dopant

# "samarium-doped ceria", "samaria", "SDC" all mean the element symbol Sm.
_DOPANT_ALIASES = {
    "samarium": "sm", "samaria": "sm", "sdc": "sm",
    "gadolinium": "gd", "gadolinia": "gd", "gdc": "gd",
    "calcium": "ca", "calcia": "ca", "cdc": "ca",
    "aluminium": "al", "aluminum": "al", "alumina": "al", "adc": "al",
    "barium": "ba", "bdc": "ba",
    "lanthanum": "la", "lanthana": "la", "ldc": "la",
    "yttrium": "y", "yttria": "y", "ydc": "y",
    "praseodymium": "pr", "pdc": "pr",
    "neodymium": "nd", "ndc": "nd",
    "strontium": "sr", "magnesium": "mg", "zinc": "zn", "bismuth": "bi",
}
# Whole-value requests for "no dopant filter". Not "na" (sodium) and not "undoped"
# (a real request that this dataset has no rows for).
_NO_FILTER = {"", "any", "all", "none", "n/a", "null", "*", "every", "all dopants", "any dopant"}
# Words that describe doping rather than name a dopant.
_DOPANT_FILLER = re.compile(
    r"\b(?:co-?doped|doped|doping|dopants?|ceria|ceo2|oxides?|carbonates?|(?:nano)?composites?|"
    r"electrolytes?|delta|mol|percent|any|all|every|with)\b|%|δ",
    re.I,
)
# English words that are also element symbols (In, At, As, B...): filler only in lower case,
# so "Sm+Ca in ceria" drops "in" but "In-doped ceria" keeps indium.
_STOPWORDS = re.compile(r"\b(?:in|at|of|for|the|on|by|as|is|a|an|to|from|be|no)\b")
# Carbonate formulas inside a dopant or carbonate value: Na2CO3, (Li/Na)2CO3, (Li,Na)2CO3, LiNaCO3.
_CARBONATE_IN_TEXT = re.compile(r"[A-Za-z0-9()/:,.\[\]]*CO3", re.I)
_UNION_WORDS = re.compile(r"\b(?:or|vs|versus|and)\b|[,;/&]", re.I)
_UNION_SPLIT = re.compile(r"\s+(?:or|vs\.?|versus|and)\s+|\s*[;&]\s*|,(?![^()]*\))\s*", re.I)
_HOST = {"ce", "o"}  # in a formula like Ce0.8Sm0.2O1.9 these are the host, not dopants


def _token_symbol(token: str) -> str | None:
    low = token.lower()
    if low in _DOPANT_ALIASES:
        return _DOPANT_ALIASES[low]
    if len(token) <= 2 and low in _ELEMENTS:
        return low
    return None


def _symbols_in(text: str) -> set[str] | None:
    """Element symbols named in ``text``, or None if any word is not an element.

    'Sm or Ca' -> {sm, ca}; 'Ce0.8Sm0.2O1.9' -> {sm} (host dropped); 'LN' -> None.
    """
    symbols: set[str] = set()
    for word in re.findall(r"[A-Za-z]+", _UNION_WORDS.sub(" ", text)):
        symbol = _token_symbol(word)
        if symbol is None and word[:1].isupper():
            parts = re.findall(r"[A-Z][a-z]?", word)  # CamelCase formula chunk: "CeSm"
            if "".join(parts) == word and all(p.lower() in _ELEMENTS for p in parts):
                symbols.update(p.lower() for p in parts)
                continue
        if symbol is None:
            return None
        symbols.add(symbol)
    symbols -= _HOST
    return symbols or None


def _row_dopants(ceria_dopant: str) -> set[str]:
    """'Sm+Ca (co-doped)' -> {'sm', 'ca'}. Symbols start with a capital, so the
    lowercase 'co' in 'co-doped' is not mistaken for cobalt."""
    return {symbol.lower() for symbol in re.findall(r"[A-Z][a-z]?", ceria_dopant)}


def _label_name(label: str) -> str:
    """The name part of a composition label: 'LN-SDC ((Li/Na)2CO3-SDC ...)' -> 'LN-SDC'.
    The parenthesised part is free-text description and must not be searched for names."""
    return re.split(r"\s[(\[]", label or "", maxsplit=1)[0]


def _names(value: str, row: dict) -> bool:
    """Does ``value`` appear as a whole name in the row's composition name, dopant or paper_id?"""
    pattern = rf"(?<![A-Za-z0-9_\-]){re.escape(value)}(?![A-Za-z0-9])"
    fields = (_label_name(row.get("composition_label", "")), row.get("ceria_dopant", ""), row.get("paper_id", ""))
    return any(re.search(pattern, field, re.I) for field in fields)


def _one_dopant(raw: str, rows: list[dict]) -> tuple[Any, str | None]:
    """Matcher for one dopant request (no union): (predicate or None for no filter, note)."""
    if raw.lower() in _NO_FILTER:
        return None, None
    without_carbonate = _CARBONATE_IN_TEXT.sub(" ", raw)
    # '+' means co-doping only between two names ('Sm+Ca'), not in 'SDC + Na2CO3'.
    co_doped = re.search(r"[A-Za-z]\s*\+\s*[A-Za-z]", without_carbonate) is not None or (
        re.search(r"\bco-?doped\b", raw, re.I) is not None
    )
    cleaned = _STOPWORDS.sub(" ", _DOPANT_FILLER.sub(" ", without_carbonate))
    cleaned = re.sub(r"[\s\-+()\[\]]+", " ", cleaned).strip()
    if not cleaned:
        if co_doped:  # 'co-doped ceria': any row with two or more dopants
            return (lambda row: len(_row_dopants(row.get("ceria_dopant", ""))) >= 2), None
        return None, None  # the value was only filler, e.g. "doped ceria"

    def by_symbols(symbols: set[str]):
        if co_doped or len(symbols) == 1:
            return lambda row: symbols <= _row_dopants(row.get("ceria_dopant", ""))
        return lambda row: bool(symbols & _row_dopants(row.get("ceria_dopant", "")))

    single = _token_symbol(raw) if re.fullmatch(r"[A-Za-z]+", raw) else None
    if single:  # 'Sm', 'SDC', 'samarium': never read as a composition name
        return by_symbols({single}), None
    if any(_names(raw, row) for row in rows):  # 'LN-SDC', 'BSDC', 'SDC-Na2CO3', 'Sm+Ca'
        return (lambda row: _names(raw, row)), None
    # 'the LNK-SDC composite', 'LN-SDC binary-carbonate composite': a hyphenated composition
    # name inside a longer phrase. Hyphenated only, so a bare 'SDC' still means samarium.
    for token in without_carbonate.split():
        token = token.strip("()[],.;:'\"")
        if "-" in token.strip("-") and any(_names(token, row) for row in rows):
            return (lambda row, name=token: _names(name, row)), None
    symbols = _symbols_in(cleaned)
    if symbols:
        return by_symbols(symbols), None
    note = (
        f"dopant {raw!r} is not an element symbol, element name or composition name in the "
        "dataset; use an element symbol such as 'Sm', or omit the dopant and filter otherwise"
    )
    return (lambda row: False), note


def _dopant_matcher(dopant: Any, rows: list[dict]) -> tuple[Any, str | None]:
    """Return (predicate or None for no filter, note for the summary).

    'Sm or Ca', 'LN-SDC, NK-SDC' and ['LN-SDC', 'N-SDC'] mean either; 'Sm+Ca' and
    'co-doped' are single requests for co-doped rows.
    """
    if isinstance(dopant, (list, tuple)):
        parts = [str(d).strip() for d in dopant if str(d).strip()]
    else:
        raw = str(dopant or "").strip()
        if raw.lower() in _NO_FILTER:
            return None, None
        parts = [p.strip() for p in _UNION_SPLIT.split(raw) if p and p.strip()] or [raw]
    if len(parts) == 1:
        return _one_dopant(parts[0], rows)
    matchers, notes = [], []
    for part in parts:
        matcher, note = _one_dopant(part, rows)
        if matcher is None:
            return None, None  # one part asks for everything
        matchers.append(matcher)
        if note:
            notes.append(note)
    return (lambda row: any(m(row) for m in matchers)), ("; ".join(notes) or None)


# ---------------------------------------------------------------------- carbonate

_CATION_NAMES = {
    "lithium": "Li", "sodium": "Na", "potassium": "K", "caesium": "Cs", "cesium": "Cs",
    "rubidium": "Rb", "barium": "Ba", "calcium": "Ca", "strontium": "Sr",
}
_CARBONATE_CATIONS = ("Li", "Na", "K", "Cs", "Rb", "Ba", "Ca", "Sr")
_YES = {
    "any", "yes", "y", "true", "1", "with", "present", "carbonate", "carbonates", "composite",
    "composites", "any carbonate", "with carbonate", "carbonate composite", "carbonate composites",
    "molten carbonate", "alkali carbonate", "alkali carbonates", "all",
}
_NO = {
    "none", "no", "n", "false", "0", "without", "absent", "single-phase", "single phase", "pure",
    "no carbonate", "without carbonate", "carbonate-free", "carbonate free", "no carbonates",
}
_SIZE_WORDS = {"single": 1, "binary": 2, "ternary": 3}


def _cations_in_cell(text: str) -> frozenset[str]:
    """Cations of every carbonate named in a dataset cell: '(Li/Na)2CO3 binary' -> {Li, Na}."""
    cations: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9()/:,.\[\]]*CO3", text or ""):
        cations.update(s for s in re.findall(r"[A-Z][a-z]?", token[:-3]) if s in _CARBONATE_CATIONS)
    return frozenset(cations)


def _cations_in_spec(text: str) -> frozenset[str]:
    """Cations named in a request, in any case: 'na2co3-k2co3', 'Sodium carbonate', 'LiNaCO3'."""
    cations: set[str] = set()
    for token in _CARBONATE_IN_TEXT.findall(text):
        body = token[:-3]
        for symbol in _CARBONATE_CATIONS:
            if re.search(rf"(?<![a-z]){symbol.lower()}(?![a-z])", body.lower()) or symbol in body:
                cations.add(symbol)
    for name, symbol in _CATION_NAMES.items():
        if re.search(rf"\b{name}\b", text, re.I):
            cations.add(symbol)
    return frozenset(cations)


def _has_carbonate(cell: str) -> bool:
    low = (cell or "").strip().lower()
    if not low or low == "none":
        return False
    return bool(_cations_in_cell(cell)) or "carbonate" in low


# List separators for carbonate requests. Not 'and' ("Na2CO3 and K2CO3" is the binary
# system) and not a comma inside parentheses ("(Li,Na)2CO3").
_CARBONATE_SPLIT = re.compile(r"\s+(?:or|vs\.?|versus)\s+|\s*;\s*|,(?![^()]*\))\s*", re.I)


def _one_carbonate(text: str) -> Any:
    low = re.sub(r"\s+", " ", text.lower()).strip()
    if not low or low in {"null", "n/a", "*"}:
        return None
    if low in _YES:
        return _has_carbonate
    if low in _NO:
        return lambda cell: not _has_carbonate(cell)
    bare = next((c for c in _CARBONATE_CATIONS if c.lower() == low), None) or _CATION_NAMES.get(low)
    if bare:  # 'Li' or 'lithium': any carbonate that contains lithium
        return lambda cell: bare in _cations_in_cell(cell)
    cations = _cations_in_spec(text)
    if cations:  # a carbonate system, 'Na2CO3' or 'Na2CO3-K2CO3': exactly those cations
        return lambda cell: _cations_in_cell(cell) == cations
    size = next((n for word, n in _SIZE_WORDS.items() if re.search(rf"\b{word}\b", low)), None)
    if size:
        return lambda cell: len(_cations_in_cell(cell)) == size
    if "carbonate" in low or "co3" in low:  # 'ceria-carbonate composite', 'no carbonate phase'
        if re.search(r"\b(?:no|not|without|absent|free)\b|-free", low):
            return lambda cell: not _has_carbonate(cell)
        return _has_carbonate
    pattern = rf"(?<![A-Za-z0-9]){re.escape(low)}(?![A-Za-z0-9])"
    return lambda cell: re.search(pattern, (cell or "").lower()) is not None


def _carbonate_matcher(spec: Any) -> Any:
    """Return a predicate over secondary_phase text, or None for no filter.
    'Na2CO3 or Li2CO3' and ['Na2CO3', 'Li2CO3'] mean either system."""
    if spec is None:
        return None
    if isinstance(spec, bool):
        return lambda cell: _has_carbonate(cell) is spec
    parts = [str(s) for s in spec] if isinstance(spec, (list, tuple)) else _CARBONATE_SPLIT.split(normalize(str(spec)))
    matchers = []
    for part in (p.strip() for p in parts if p and p.strip()):
        matcher = _one_carbonate(normalize(part))
        if matcher is None:
            return None
        matchers.append(matcher)
    if not matchers:
        return None
    return matchers[0] if len(matchers) == 1 else (lambda cell: any(m(cell) for m in matchers))


# ---------------------------------------------------------------------- temperature

_ABOVE = re.compile(
    r"\b(?:above|over|greater than|more than|higher than|at least|exceeding|from|min(?:imum)?)\b"
    r"|\bor (?:higher|above|more|greater)\b|>=?|\d\s*\+\s*$",
    re.I,
)
_BELOW = re.compile(
    r"\b(?:below|under|less than|lower than|at most|up to|max(?:imum)?)\b|\bor (?:lower|below|less)\b|<=?",
    re.I,
)
_TOLERANCE = re.compile(
    r"^(?P<center>.*?\d(?:\s*°?\s*(?:C|K)\b)?)\s*(?:\+/-|\+-|plus or minus)\s*(?P<spread>\d+(?:\.\d+)?)", re.I
)


def _celsius(text: str) -> list[float]:
    """Temperatures written in ``text``, in Celsius (numbers marked K are converted)."""
    return [
        value - 273.15 if unit == "K" else value
        for value, kind, unit in quantities(text)
        if kind in ("temperature", "bare")
    ]


def _temperature_window(value: Any, tolerance: float) -> tuple[float, float] | str:
    """(low, high) in Celsius, or an error message for the model."""
    if isinstance(value, bool):
        return f"temperature_C must be a number in Celsius, got {value!r}"
    if isinstance(value, (int, float)):
        return float(value) - tolerance, float(value) + tolerance
    if isinstance(value, (list, tuple)):
        value = " to ".join(str(v) for v in value)
    text = normalize(str(value))
    spread = _TOLERANCE.search(text)
    if spread:  # '600 +/- 25', '600 C plus or minus 20'
        centers = _celsius(spread.group("center"))
        if len(centers) == 1:
            return centers[0] - float(spread.group("spread")), centers[0] + float(spread.group("spread"))
    main = re.sub(r"\([^)]*\)", " ", text)  # '600 C (873 K)': the bracket repeats the value
    values = _celsius(main)
    if not values:
        return f"temperature_C must be a number in Celsius, got {value!r}"
    if len(values) > 2:
        return f"temperature_C takes one temperature or a range of two, got {value!r}"
    if len(values) == 2:
        return min(values), max(values)
    if _ABOVE.search(main):
        return values[0], math.inf
    if _BELOW.search(main):
        return -math.inf, values[0]
    return values[0] - tolerance, values[0] + tolerance


# ---------------------------------------------------------------------- rows

def _num(value: Any) -> float | None:
    """Strict parse for dataset cells: a cell is a number or it is not."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _load_rows() -> tuple[list[dict], tuple[str, ...]]:
    path = config.dataset_csv()
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found at {path} - check paths.dataset_csv in config.yaml"
        )
    # utf-8-sig: the CSV comes out of spreadsheet tooling and may carry a BOM.
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = tuple(reader.fieldnames or ())
        rows = [{(key or ""): (val or "") for key, val in row.items()} for row in reader]
    return rows, columns


def _shape_row(row: dict, fields: tuple[str, ...]) -> dict:
    out: dict[str, Any] = {}
    for field in fields:
        raw = row.get(field, "")
        if field in NUMERIC_FIELDS:
            out[field] = _num(raw)
        else:
            out[field] = raw.strip() or None
    return out


def _best(rows: list[dict], value_field: str, temp_field: str) -> dict | None:
    candidates = [r for r in rows if r.get(value_field) is not None]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r[value_field])
    return {
        "paper_id": best.get("paper_id"),
        "composition_label": best.get("composition_label"),
        value_field: best[value_field],
        temp_field: best.get(temp_field),
    }


def query_dataset(
    dopant: str | None = None,
    carbonate: Any = None,
    temperature_C: Any = None,
    paper_id: str | None = None,
    limit: int | None = None,
    temperature_tolerance_C: float | None = None,
    temperature_field: str = "conductivity_temp_C",
    full_rows: bool = False,
) -> dict:
    """Filter the extracted dataset and return the matching rows.

    Args:
        dopant: element symbol or name, several of them, a composition name or a formula.
        carbonate: "any"/"none", a carbonate system ("Na2CO3", "(Li/Na)2CO3"), or
            "single"/"binary"/"ternary".
        temperature_C: one temperature in Celsius (matched within the tolerance), a kelvin
            value ("873 K") or a range ("500-600", "above 550").
        paper_id: restrict to one source paper.
        limit: maximum rows to return (capped by ``dataset.max_limit``).
        temperature_tolerance_C: half-window for a single temperature (default from config).
        temperature_field: which temperature the filter refers to: conductivity (default),
            peak power or sintering.
        full_rows: return every column instead of the curated subset.

    Returns:
        A JSON-serialisable dict with ``ok``, a one-line ``summary``, the filters that were
        applied, aggregate statistics, and the matching ``rows`` (each carrying ``paper_id``).
    """
    try:
        rows, columns = _load_rows()
    except Exception as exc:
        return {"ok": False, "tool": "query_dataset", "error_code": "dataset_unavailable", "error": str(exc)}

    if temperature_field not in TEMPERATURE_FIELDS:
        return {
            "ok": False,
            "tool": "query_dataset",
            "error": (
                f"temperature_field must be one of {list(TEMPERATURE_FIELDS)}, "
                f"got {temperature_field!r}"
            ),
        }

    tolerance = (
        float(config.get("dataset.default_temperature_tolerance_C", 25.0))
        if temperature_tolerance_C is None
        else float(temperature_tolerance_C)
    )
    max_limit = int(config.get("dataset.max_limit", 100))
    try:
        limit_eff = int(config.get("dataset.default_limit", 25)) if limit is None else int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "tool": "query_dataset", "error": f"limit must be an integer, got {limit!r}"}
    limit_eff = max(1, min(limit_eff, max_limit))

    window = None
    if temperature_C not in (None, ""):
        window = _temperature_window(temperature_C, tolerance)
        if isinstance(window, str):
            return {"ok": False, "tool": "query_dataset", "error": window}

    dopant_ok, note = _dopant_matcher(dopant, rows)
    carbonate_ok = _carbonate_matcher(carbonate)
    paper_q = str(paper_id or "").strip().strip("[]()'\"").strip().lower()  # "[ref11_LN-SDC]" too
    if paper_q in _NO_FILTER:
        paper_q = ""  # 'none', 'null', 'any' from a model mean no paper filter

    def in_window(row: dict) -> bool:
        value = _num(row.get(temperature_field))
        return value is not None and window[0] <= value <= window[1]

    checks: list[tuple[str, Any]] = []  # (label, predicate) for each active filter
    if dopant_ok is not None:
        checks.append((f"dopant {dopant!r}", dopant_ok))
    if carbonate_ok is not None:
        checks.append((f"carbonate {carbonate!r}", lambda row: carbonate_ok(row.get("secondary_phase", ""))))
    if window is not None:
        checks.append(("the temperature", in_window))
    if paper_q:
        checks.append((f"paper_id {paper_id!r}", lambda row: paper_q in row.get("paper_id", "").lower()))
    matched = [row for row in rows if all(check(row) for _, check in checks)]

    near_miss = None
    if not matched and len(checks) > 1:  # say which filter emptied the result, so the model can relax it
        near_miss = "; ".join(
            f"{label} alone matches {sum(1 for row in rows if check(row))} row(s)" for label, check in checks
        )

    fields = columns if full_rows else DEFAULT_FIELDS
    shaped = [_shape_row(row, tuple(fields)) for row in matched]
    measured = [_shape_row(row, DEFAULT_FIELDS) for row in matched]

    conductivities = [r["conductivity_S_cm"] for r in measured if r.get("conductivity_S_cm") is not None]
    stats: dict[str, Any] = {
        "n_matches": len(matched),
        "n_papers": len({row.get("paper_id", "") for row in matched}),
        "n_with_conductivity": len(conductivities),
        "n_with_pmax": sum(1 for r in measured if r.get("pmax_mW_cm2") is not None),
    }
    if conductivities:
        stats["conductivity_S_cm_min"] = min(conductivities)
        stats["conductivity_S_cm_max"] = max(conductivities)
        stats["best_conductivity"] = _best(measured, "conductivity_S_cm", "conductivity_temp_C")
    best_pmax = _best(measured, "pmax_mW_cm2", "pmax_temp_C")
    if best_pmax:
        stats["best_pmax"] = best_pmax

    temperature_filter = None
    if window is not None:
        low, high = window
        if math.isinf(high):
            temperature_filter = f">= {low:g} C"
        elif math.isinf(low):
            temperature_filter = f"<= {high:g} C"
        else:
            temperature_filter = f"{low:g}-{high:g} C"
    filters = {
        "dopant": dopant or None,
        "carbonate": carbonate if carbonate not in (None, "") else None,
        "temperature_C": temperature_filter,
        "temperature_field": temperature_field if window is not None else None,
        "paper_id": paper_id or None,
    }
    active = {key: val for key, val in filters.items() if val is not None}

    if not matched:
        summary = f"No rows match {active or 'no filters'} (dataset has {len(rows)} rows)."
        if near_miss:
            summary += f" Separately: {near_miss}. Remove the filter that excludes everything and ask again."
        if window is not None and window[0] > 1000:
            summary += " temperature_C is in Celsius; that value looks like kelvin."
    else:
        summary = (
            f"{len(matched)} of {len(rows)} rows match {active or 'no filters'}, "
            f"from {stats['n_papers']} paper(s)"
        )
        best = stats.get("best_conductivity")
        if best:
            summary += (
                f"; highest conductivity {best['conductivity_S_cm']} S/cm at "
                f"{best['conductivity_temp_C']} C ({best['composition_label']}, {best['paper_id']})"
            )
        summary += "."
    if note:
        summary += f" Note: {note}."

    return {
        "ok": True,
        "tool": "query_dataset",
        "dataset_rows_total": len(rows),
        "filters_applied": active,
        "statistics": stats,
        "returned_rows": min(len(shaped), limit_eff),
        "truncated": len(shaped) > limit_eff,
        "summary": summary,
        "rows": shaped[:limit_eff],
    }
