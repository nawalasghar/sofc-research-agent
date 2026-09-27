"""Tool: materials_project - look up a formula in the free Materials Project REST API.

Returns thermodynamic stability (distance above the convex hull) and the DFT band gap for
the most stable entry matching a formula, plus the other polymorphs it found.

The API key is read from ``.env`` (``MP_API_KEY``), which is gitignored; a free key comes
from https://profile.materialsproject.org/ after registering at materialsproject.org.

Note for doped compositions: Materials Project indexes ordered DFT structures, so a query
for ``Ce0.8Sm0.2O2`` will not match. Look up the parent phases instead - ``CeO2``, ``Sm2O3``,
``Na2CO3`` - and reason about the composite from those.
"""
from __future__ import annotations

from typing import Any

from .. import config

# Requested explicitly: the summary endpoint returns ~69 fields per entry otherwise.
SUMMARY_FIELDS = (
    "material_id",
    "formula_pretty",
    "chemsys",
    "is_stable",
    "energy_above_hull",
    "formation_energy_per_atom",
    "band_gap",
    "is_gap_direct",
    "is_metal",
    "symmetry",
    "density",
    "nsites",
    "theoretical",
)

MP_WEBSITE = "https://materialsproject.org/materials"


def _round(value: Any, digits: int) -> float | None:
    return None if value is None else round(float(value), digits)


def _entry(doc: dict) -> dict:
    """Flatten one Materials Project summary document into a compact record."""
    symmetry = doc.get("symmetry") or {}
    material_id = doc.get("material_id")
    return {
        "material_id": material_id,
        "formula_pretty": doc.get("formula_pretty"),
        "chemsys": doc.get("chemsys"),
        "crystal_system": symmetry.get("crystal_system"),
        "spacegroup_symbol": symmetry.get("symbol"),
        "spacegroup_number": symmetry.get("number"),
        "is_stable": doc.get("is_stable"),
        "energy_above_hull_eV_per_atom": _round(doc.get("energy_above_hull"), 4),
        "formation_energy_eV_per_atom": _round(doc.get("formation_energy_per_atom"), 4),
        "band_gap_eV": _round(doc.get("band_gap"), 4),
        "is_gap_direct": doc.get("is_gap_direct"),
        "is_metal": doc.get("is_metal"),
        "density_g_cm3": _round(doc.get("density"), 3),
        "n_sites": doc.get("nsites"),
        "theoretical": doc.get("theoretical"),
        "mp_url": f"{MP_WEBSITE}/{material_id}" if material_id else None,
    }


def _describe_stability(entry: dict) -> str:
    above_hull = entry.get("energy_above_hull_eV_per_atom")
    if entry.get("is_stable"):
        return "on the convex hull - thermodynamically stable at 0 K"
    if above_hull is None:
        return "stability unknown (no hull energy returned)"
    return f"metastable - {above_hull * 1000:.0f} meV/atom above the convex hull"


def _describe_gap(entry: dict) -> str:
    gap = entry.get("band_gap_eV")
    if gap is None:
        return "band gap not reported"
    if gap <= 0:
        return "no band gap (metallic in DFT)"
    kind = "direct" if entry.get("is_gap_direct") else "indirect"
    return f"{gap:.3f} eV {kind} band gap (DFT/GGA, typically underestimated)"


def parse_summary_response(payload: dict, formula: str) -> dict:
    """Turn a raw API payload into the tool result. Pure, so it is testable offline."""
    docs = payload.get("data") or []
    entries = [_entry(doc) for doc in docs if isinstance(doc, dict)]
    # Ask for hull-sorted results, but do not trust the order.
    entries.sort(key=lambda e: (e["energy_above_hull_eV_per_atom"] is None,
                                e["energy_above_hull_eV_per_atom"] or 0.0))

    if not entries:
        return {
            "ok": True,
            "tool": "materials_project",
            "formula_query": formula,
            "found": False,
            "n_candidates": 0,
            "summary": (
                f"No Materials Project entry matches formula {formula!r}. Doped or composite "
                "compositions are not indexed - try the parent phase (e.g. CeO2, Sm2O3, Na2CO3)."
            ),
            "most_stable": None,
            "polymorphs": [],
        }

    best = entries[0]
    stability = _describe_stability(best)
    gap = _describe_gap(best)
    summary = (
        f"{best['formula_pretty']} ({best['material_id']}, {best['spacegroup_symbol']} "
        f"{best['crystal_system']}): {stability}; {gap}. "
        f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'} returned for {formula!r}."
    )

    return {
        "ok": True,
        "tool": "materials_project",
        "formula_query": formula,
        "found": True,
        "n_candidates": len(entries),
        "summary": summary,
        "stability": stability,
        "band_gap_description": gap,
        "most_stable": best,
        "polymorphs": [
            {
                "material_id": e["material_id"],
                "formula_pretty": e["formula_pretty"],
                "spacegroup_symbol": e["spacegroup_symbol"],
                "energy_above_hull_eV_per_atom": e["energy_above_hull_eV_per_atom"],
                "band_gap_eV": e["band_gap_eV"],
                "is_stable": e["is_stable"],
            }
            for e in entries[1:]
        ],
    }


def materials_project(formula: str, limit: int | None = None) -> dict:
    """Look up a chemical formula in the Materials Project database.

    Args:
        formula: chemical formula of an ordered phase, e.g. "CeO2", "Sm2O3", "Na2CO3".
        limit: how many polymorphs to fetch before picking the most stable one.

    Returns:
        A JSON-serialisable dict with ``ok``, a one-line ``summary``, and - when a match is
        found - ``most_stable`` (stability, band gap, symmetry, density) plus ``polymorphs``.
    """
    formula = (formula or "").strip()
    if not formula:
        return {"ok": False, "tool": "materials_project", "error": "formula must not be empty"}

    key_var = str(config.get("materials_project.api_key_env", "MP_API_KEY"))
    api_key = config.secret(key_var)
    if not api_key:
        return {
            "ok": False,
            "tool": "materials_project",
            "error_code": "missing_api_key",
            "error": (
                f"no {key_var} found. Copy .env.example to .env and paste a free key from "
                "https://profile.materialsproject.org/ (register at materialsproject.org)."
            ),
        }

    base_url = str(config.get("materials_project.base_url", "https://api.materialsproject.org")).rstrip("/")
    timeout = float(config.get("materials_project.timeout_s", 30))
    try:
        limit_eff = int(config.get("materials_project.candidate_limit", 10)) if limit is None else int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "tool": "materials_project", "error": f"limit must be an integer, got {limit!r}"}
    limit_eff = max(1, min(limit_eff, 100))

    import requests  # deferred so the offline tools do not need it

    params = {
        "formula": formula,
        "_fields": ",".join(SUMMARY_FIELDS),
        "_sort_fields": "energy_above_hull",
        "_limit": limit_eff,
    }
    try:
        response = requests.get(
            f"{base_url}/materials/summary/",
            params=params,
            headers={"X-API-KEY": api_key, "Accept": "application/json"},
            timeout=timeout,
        )
    except Exception as exc:  # connection, DNS, timeout
        return {
            "ok": False,
            "tool": "materials_project",
            "error_code": "network_error",
            "error": f"could not reach {base_url}: {exc}",
        }

    if response.status_code in (401, 403):
        return {
            "ok": False,
            "tool": "materials_project",
            "error_code": "unauthorized",
            "error": f"Materials Project rejected the API key (HTTP {response.status_code}). Check {key_var} in .env.",
        }
    if response.status_code == 429:
        return {
            "ok": False,
            "tool": "materials_project",
            "error_code": "rate_limited",
            "error": "Materials Project rate limit hit (HTTP 429); wait and retry.",
        }
    if response.status_code >= 400:
        detail = response.text[:300].replace("\n", " ")
        return {
            "ok": False,
            "tool": "materials_project",
            "error_code": f"http_{response.status_code}",
            "error": f"HTTP {response.status_code} from Materials Project: {detail}",
        }

    try:
        payload = response.json()
    except ValueError:
        return {
            "ok": False,
            "tool": "materials_project",
            "error_code": "bad_json",
            "error": "Materials Project returned a non-JSON body",
        }

    return parse_summary_response(payload, formula)
