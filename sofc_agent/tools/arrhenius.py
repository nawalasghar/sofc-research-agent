"""Tool: arrhenius - fit an activation energy from conductivity-versus-temperature data.

Two conventions appear in the SOFC literature, so both are supported:

* ``sigma_T`` (default, usual for oxide-ion conductors)
      sigma = (A / T) * exp(-Ea / (kB * T))   ->   ln(sigma * T) is linear in 1/T
* ``sigma``
      sigma = A * exp(-Ea / (kB * T))         ->   ln(sigma) is linear in 1/T

Either way the slope of the straight line is -Ea / kB, so Ea = -slope * kB. The fit is a
closed-form ordinary least squares, so this tool needs nothing beyond the standard library.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Sequence

from ..numbers import _UNIT_AFTER, CONDUCTIVITY_FACTORS, NUMBER, normalize, parse_quantity

# CODATA 2018 Boltzmann constant in eV/K, and the eV -> kJ/mol conversion.
KB_EV_PER_K = 8.617333262e-5
EV_TO_KJ_PER_MOL = 96.48533212

_MODEL_ALIASES = {
    "sigma_t": "sigma_T",
    "sigmat": "sigma_T",
    "sigma*t": "sigma_T",
    "st": "sigma_T",
    "ln(sigma*t)": "sigma_T",
    "sigma": "sigma",
    "s": "sigma",
    "ln(sigma)": "sigma",
}

_UNIT_ALIASES = {
    "c": "C", "celsius": "C", "degc": "C", "deg_c": "C", "\u00b0c": "C",
    "k": "K", "kelvin": "K",
}


# One list item written as text: a number (any notation) and an optional unit.
_ITEM = re.compile(rf"(?:{NUMBER.pattern})(?:{_UNIT_AFTER.pattern})?")


def _split_text_list(text: str) -> list[Any]:
    """Parse '[400, 450]', '400, 450', '400 450' or '773 K 823 K' - a small model sends all of them."""
    text = normalize(text).strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        return parsed
    body = text.strip("[]() ")
    parts = [p.strip() for p in re.split(r"[,;]", body) if p.strip()]
    if len(parts) == 1:  # no separators: pick the items out one by one
        parts = [m.group().strip() for m in _ITEM.finditer(body)] or parts
    return parts


def _as_list(values: Any, label: str) -> list[Any]:
    if isinstance(values, bytes):
        values = values.decode("utf-8", errors="replace")
    if isinstance(values, str):
        values = _split_text_list(values)
    if values is None or isinstance(values, dict) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a list of numbers, got {type(values).__name__}")
    return list(values)


def _parse_items(values: Any, label: str, kind: str) -> list[tuple[float, str | None]]:
    parsed = []
    for i, item in enumerate(_as_list(values, label)):
        try:
            parsed.append(parse_quantity(item, kind))
        except ValueError:
            raise ValueError(f"{label}[{i}] is not a {kind}: {item!r}") from None
    return parsed


def _list_units(units: list[str | None], label: str) -> list[str | None]:
    """A unit on every item, or on none, is clear. A unit written once, after the last
    item ("773, 823, 873 K") or on the first ("773 K, 823, 873"), belongs to the whole
    list. Any other mix is ambiguous and refused rather than guessed."""
    if all(u is None for u in units) or all(u is not None for u in units):
        return units
    written = [u for u in units if u is not None]
    if len(written) == 1 and (units[-1] is not None or units[0] is not None):
        return [written[0]] * len(units)
    raise ValueError(f"{label}: write the unit on every value, or once for the whole list")


def parse_temperatures(values: Any, default_unit: str) -> tuple[list[float], list[float]]:
    """(values as written, values in kelvin). A unit written on the items ("773 K", or
    "773, 823, 873 K") overrides ``default_unit``, so a model that forgets
    temperature_unit="K" still gets the right fit."""
    parsed = _parse_items(values, "temperatures", "temperature")
    units = _list_units([u for _, u in parsed], "temperatures")
    written = [value for value, _ in parsed]
    kelvin = [v + 273.15 if (u or default_unit) == "C" else v for v, u in zip(written, units)]
    return written, kelvin


def parse_conductivities(values: Any) -> list[float]:
    """Conductivities in S/cm; items may carry S/cm, S cm-1, mS/cm or S/m."""
    parsed = _parse_items(values, "conductivities", "conductivity")
    written_units = [u for _, u in parsed]
    units = _list_units(written_units, "conductivities")
    out = []
    for (value, own), unit in zip(parsed, units):
        # parse_quantity already converted items that carried a unit; a unit carried over
        # from the end of the list still has to be applied.
        out.append(value * CONDUCTIVITY_FACTORS[unit] if own is None and unit else value)
    return out


def _linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float | None]:
    """Ordinary least squares. Returns (slope, intercept, r_squared)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_squared = None if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return slope, intercept, r_squared


def arrhenius(
    temperatures: Sequence[float],
    conductivities: Sequence[float],
    temperature_unit: str = "C",
    model: str = "sigma_T",
) -> dict:
    """Fit an Arrhenius activation energy to conductivity data.

    Args:
        temperatures: measurement temperatures, one per conductivity value.
        conductivities: conductivities in S/cm (any consistent unit works; only the slope matters).
        temperature_unit: "C" (default) or "K".
        model: "sigma_T" for ln(sigma*T) vs 1/T (default), or "sigma" for ln(sigma) vs 1/T.

    Returns:
        A JSON-serialisable dict with ``ok``, a one-line ``summary``, ``activation_energy_eV``,
        the fit statistics, and the transformed points that were fitted.
    """
    unit = _UNIT_ALIASES.get(str(temperature_unit).strip().lower())
    if unit is None:
        return {
            "ok": False,
            "tool": "arrhenius",
            "error": f"temperature_unit must be 'C' or 'K', got {temperature_unit!r}",
        }

    model_key = _MODEL_ALIASES.get(str(model).strip().lower())
    if model_key is None:
        return {
            "ok": False,
            "tool": "arrhenius",
            "error": f"model must be 'sigma_T' or 'sigma', got {model!r}",
        }

    try:
        temps, temps_k = parse_temperatures(temperatures, unit)
        sigmas = parse_conductivities(conductivities)
    except ValueError as exc:
        return {"ok": False, "tool": "arrhenius", "error": str(exc)}

    if len(temps) != len(sigmas):
        return {
            "ok": False,
            "tool": "arrhenius",
            "error": (
                f"temperatures and conductivities must be the same length "
                f"({len(temps)} vs {len(sigmas)})"
            ),
        }
    if len(temps) < 2:
        return {
            "ok": False,
            "tool": "arrhenius",
            "error": f"need at least 2 points to fit a slope, got {len(temps)}",
        }

    for i, (t_k, sigma) in enumerate(zip(temps_k, sigmas)):
        if t_k <= 0:
            return {
                "ok": False,
                "tool": "arrhenius",
                "error": f"temperatures[{i}] is {temps[i]:g} ({t_k:.2f} K); must be above 0 K",
            }
        if sigma <= 0:
            return {
                "ok": False,
                "tool": "arrhenius",
                "error": f"conductivities[{i}] is {sigma!r}; conductivity must be positive to take a log",
            }
    if len(set(temps_k)) < 2:
        return {
            "ok": False,
            "tool": "arrhenius",
            "error": "need at least 2 distinct temperatures; all values are the same",
        }

    xs = [1.0 / t for t in temps_k]
    ys = [
        math.log(sigma * t) if model_key == "sigma_T" else math.log(sigma)
        for sigma, t in zip(sigmas, temps_k)
    ]

    slope, intercept, r_squared = _linear_fit(xs, ys)
    ea_ev = -slope * KB_EV_PER_K

    points = [
        {
            "temperature_C": round(t_k - 273.15, 2),
            "temperature_K": round(t_k, 2),
            "conductivity_S_cm": sigma,
            "inv_T_per_K": x,
            "y": y,
            "residual": y - (slope * x + intercept),
        }
        for t_k, sigma, x, y in zip(temps_k, sigmas, xs, ys)
    ]

    y_label = "ln(sigma*T)" if model_key == "sigma_T" else "ln(sigma)"
    prefactor_units = "S*K/cm" if model_key == "sigma_T" else "S/cm"

    notes: list[str] = []
    if ea_ev < 0:
        notes.append(
            "Negative Ea: conductivity falls with temperature in this data, which is not "
            "Arrhenius behaviour - check the temperature/conductivity pairing."
        )
    if len(temps) == 2:
        notes.append("Only 2 points: the fit is exact by construction, so r^2 carries no information.")
    elif r_squared is not None and r_squared < 0.99:
        notes.append(
            f"r^2 = {r_squared:.4f}; the points deviate from a single straight line. A composite "
            "often shows two regimes (a knee near the carbonate transition), so consider fitting "
            "the low- and high-temperature branches separately."
        )

    summary = (
        f"Ea = {ea_ev:.4f} eV ({ea_ev * EV_TO_KJ_PER_MOL:.1f} kJ/mol) from {len(temps)} points "
        f"over {min(temps_k) - 273.15:.0f}-{max(temps_k) - 273.15:.0f} C, "
        f"{y_label} vs 1/T"
        + (f", r^2 = {r_squared:.4f}" if r_squared is not None else "")
        + "."
    )

    return {
        "ok": True,
        "tool": "arrhenius",
        "model": model_key,
        "equation": (
            "sigma = (A/T)*exp(-Ea/(kB*T))" if model_key == "sigma_T" else "sigma = A*exp(-Ea/(kB*T))"
        ),
        "fitted_axes": f"{y_label} vs 1/T(K)",
        "n_points": len(temps),
        "temperature_range_C": [round(min(temps_k) - 273.15, 2), round(max(temps_k) - 273.15, 2)],
        "activation_energy_eV": round(ea_ev, 4),
        "activation_energy_kJ_per_mol": round(ea_ev * EV_TO_KJ_PER_MOL, 2),
        "slope_K": slope,
        "intercept": intercept,
        "prefactor_A": math.exp(intercept),
        "prefactor_units": prefactor_units,
        "r_squared": None if r_squared is None else round(r_squared, 6),
        "notes": notes,
        "summary": summary,
        "points": points,
    }
