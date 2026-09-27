"""Tool: query_dataset - structured questions over the extracted SOFC dataset.

The dataset is one row per reported composition/measurement, hand-extracted from the
papers, with ``paper_id`` linking every row back to its source. Filters are deliberately
forgiving because the source columns are free text: ``dopant="Sm"`` also matches the
co-doped ``Ba+Sm`` and ``Sm+Ca (co-doped)`` entries, and ``carbonate="Na2CO3"`` matches
``Na2CO3 (amorphous shell)``.

Only the standard library is used, so this tool imports instantly.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from typing import Any

from .. import config

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

# Free-text secondary_phase values are matched for a carbonate on these markers.
_CARBONATE_MARKERS = ("co3", "carbonate")
_YES = {"any", "yes", "y", "true", "1", "with", "present", "carbonate", "composite"}
_NO = {"none", "no", "n", "false", "0", "without", "absent", "single-phase", "single phase", "pure"}


def _num(value: Any) -> float | None:
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


def _has_carbonate(secondary_phase: str) -> bool:
    low = secondary_phase.strip().lower()
    if not low or low == "none":
        return False
    return any(marker in low for marker in _CARBONATE_MARKERS)


def _carbonate_matches(spec: Any, secondary_phase: str) -> bool:
    if isinstance(spec, bool):
        return _has_carbonate(secondary_phase) is spec
    text = str(spec).strip().lower()
    if text in _YES:
        return _has_carbonate(secondary_phase)
    if text in _NO:
        return not _has_carbonate(secondary_phase)
    return text in secondary_phase.lower()


def _shape_row(row: dict, fields: tuple[str, ...]) -> dict:
    out: dict[str, Any] = {}
    for field in fields:
        raw = row.get(field, "")
        if field in NUMERIC_FIELDS:
            out[field] = _num(raw)
        else:
            out[field] = raw.strip() or None
    return out


def query_dataset(
    dopant: str | None = None,
    carbonate: Any = None,
    temperature_C: float | None = None,
    paper_id: str | None = None,
    limit: int | None = None,
    temperature_tolerance_C: float | None = None,
    temperature_field: str = "conductivity_temp_C",
    full_rows: bool = False,
) -> dict:
    """Filter the extracted dataset and return the matching rows.

    Args:
        dopant: ceria dopant element, matched as a substring ("Sm", "Ca", "Al", "Ba").
        carbonate: "any"/"none" for presence, or a phase substring ("Na2CO3", "ternary").
        temperature_C: measurement temperature to match, within ``temperature_tolerance_C``.
        paper_id: restrict to one source paper.
        limit: maximum rows to return (capped by ``dataset.max_limit``).
        temperature_tolerance_C: half-window for the temperature match (default from config).
        temperature_field: which temperature column to match against.
        full_rows: return every column instead of the curated subset.

    Returns:
        A JSON-serialisable dict with ``ok``, a one-line ``summary``, the filters that were
        applied, aggregate statistics, and the matching ``rows`` (each carrying ``paper_id``).
    """
    try:
        rows, columns = _load_rows()
    except Exception as exc:
        return {"ok": False, "tool": "query_dataset", "error": str(exc)}

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

    target_temp = _num(temperature_C)
    if temperature_C not in (None, "") and target_temp is None:
        return {
            "ok": False,
            "tool": "query_dataset",
            "error": f"temperature_C must be a number, got {temperature_C!r}",
        }

    dopant_q = (dopant or "").strip().lower()
    paper_q = (paper_id or "").strip().lower()

    matched: list[dict] = []
    for row in rows:
        if dopant_q and dopant_q not in row.get("ceria_dopant", "").lower():
            continue
        if carbonate is not None and not _carbonate_matches(carbonate, row.get("secondary_phase", "")):
            continue
        if target_temp is not None:
            value = _num(row.get(temperature_field))
            if value is None or abs(value - target_temp) > tolerance:
                continue
        if paper_q and paper_q not in row.get("paper_id", "").lower():
            continue
        matched.append(row)

    fields = columns if full_rows else DEFAULT_FIELDS
    shaped = [_shape_row(row, tuple(fields)) for row in matched]

    conductivities = [
        (row["conductivity_S_cm"], row) for row in shaped if row.get("conductivity_S_cm") is not None
    ]
    stats: dict[str, Any] = {
        "n_matches": len(matched),
        "n_papers": len({row.get("paper_id", "") for row in matched}),
        "n_with_conductivity": len(conductivities),
    }
    if conductivities:
        best_value, best_row = max(conductivities, key=lambda pair: pair[0])
        values = [value for value, _ in conductivities]
        stats.update({
            "conductivity_S_cm_min": min(values),
            "conductivity_S_cm_max": max(values),
            "best_conductivity": {
                "paper_id": best_row.get("paper_id"),
                "composition_label": best_row.get("composition_label"),
                "conductivity_S_cm": best_value,
                "conductivity_temp_C": best_row.get("conductivity_temp_C"),
            },
        })

    filters = {
        "dopant": dopant or None,
        "carbonate": carbonate if carbonate is not None else None,
        "temperature_C": target_temp,
        "temperature_field": temperature_field if target_temp is not None else None,
        "temperature_tolerance_C": tolerance if target_temp is not None else None,
        "paper_id": paper_id or None,
    }
    active = {key: val for key, val in filters.items() if val is not None}

    if not matched:
        summary = f"No rows match {active or 'no filters'} (dataset has {len(rows)} rows)."
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
