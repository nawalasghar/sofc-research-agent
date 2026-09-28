"""Reading numbers the way people and small models write them.

One parser shared by the Arrhenius tool, the agent's grounding checks and the evaluation,
so all three agree on what "the same number" means:

* scientific notation in any common spelling: ``1.3e-2``, ``1.3 x 10^-2``, ``1.3*10**-2``,
  ``1.3 x 10^{-2}``, the caret-less ``1.3x10-2``, and the typeset times sign with a
  superscript exponent;
* numbers glued to letters are not numbers (the 2 in CeO2, the 0.8 in Ce0.8Sm0.2O2,
  the 17 in p.17), so chemical formulas and page references never count as values;
* a unit written once after a list applies to the whole list ("0.02, 0.05 and 0.11 S/cm");
* rounding comparisons follow how people report values (0.8065 may be written 0.807 or
  0.806, 5.057 may be written 5.06 or 5.1), with at least two significant digits required.
"""
from __future__ import annotations

import math
import re
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation

_SUPERSCRIPTS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
})
_SUPERSCRIPT_CHARS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺"
_DASHES = str.maketrans({"−": "-", "–": "-", "‒": "-", "‐": "-"})
_TIMES = ("\\times", "\\cdot", "×", "·", "⋅")  # LaTeX, times sign, middle dots

_MANTISSA = r"[-+]?(?:\d+\.?\d*|\.\d+)"
# "10^-2", "10^(-2)", and the caret-less "10-2" (sign then mandatory) people type in a hurry.
_EXPONENT = r"(?:\^\s*\(?\s*[-+]?\s*\d{1,3}\s*\)?|\(?\s*[-+]\s*\d{1,3}\s*\)?)"
_SCI = rf"{_MANTISSA}\s*[xX*]\s*10\s*{_EXPONENT}"
_POWER = r"10\s*\^\s*\(?\s*[-+]?\s*\d{1,3}\s*\)?"
# At most two exponent digits: PDF text renders "600-800" as "600e800", which is a range.
_EXP = rf"{_MANTISSA}[eE][-+]?\d{{1,2}}(?!\d)"
# A value must not be glued to a preceding letter, digit, underscore, dot or closing
# bracket: the 2 in "CeO2", "p.17" and "(Li/Na)2CO3" are not values.
NUMBER = re.compile(rf"(?<![A-Za-z0-9_.)\]])(?:{_SCI}|{_POWER}|{_EXP}|{_MANTISSA})")
ANY_DIGITS = re.compile(r"\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """ASCII spelling: typeset scientific notation becomes '1.3 x 10^-2'; unicode minus
    and dashes become '-'; the signs >= and <= become '>=' and '<='."""
    t = str(text)
    for sign in _TIMES:
        t = t.replace(sign, " x ")
    t = re.sub(rf"10\s*(?=[{_SUPERSCRIPT_CHARS}])", "10^", t)
    t = t.translate(_SUPERSCRIPTS).translate(_DASHES)
    t = t.replace("≥", ">=").replace("≤", "<=").replace("±", "+/-")
    t = t.replace("’", "'").replace("{", "").replace("}", "").replace("**", "^")
    # Unit exponents: 'S cm-1', 'S cm^-1', 'S x cm-1' (from S.cm with a superscript -1)
    # become 'S/cm', so the '1' is not read as a number and the unit is recognised.
    t = re.sub(r"(?<![A-Za-z])([mM]?S)\s*(?:[.x*]\s*)?cm\s*\^?\s*\(?\s*-\s*1(?:\s*\))?(?!\d)", r"\1/cm", t)
    return re.sub(r"(?<![A-Za-z])(S)\s*(?:[.x*]\s*)?m\s*\^?\s*\(?\s*-\s*1(?:\s*\))?(?!\d)", r"\1/m", t)


def to_float(token: str) -> float:
    """Value of one token matched by NUMBER (after normalize). Raises ValueError for a
    value that is not a finite float (e.g. 10^400)."""
    t = re.sub(r"\s+", "", token)
    try:
        sci = re.fullmatch(rf"({_MANTISSA})[xX*]10\^?\(?([-+]?\d+)\)?", t)
        if sci:
            value = float(sci.group(1)) * 10.0 ** int(sci.group(2))
        else:
            power = re.fullmatch(r"10\^\(?([-+]?\d+)\)?", t)
            value = 10.0 ** int(power.group(1)) if power else float(t)
    except OverflowError:
        raise ValueError(f"number out of range: {token!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"number out of range: {token!r}")
    return value


def _finite_matches(text: str) -> list[tuple[re.Match, float]]:
    found = []
    for match in NUMBER.finditer(text):
        try:
            found.append((match, to_float(match.group())))
        except ValueError:
            continue  # garbled or absurd: not a value anyone can rely on
    return found


def numbers_in(text: str) -> list[float]:
    """Every standalone finite value in ``text``, scientific notation included."""
    return [value for _, value in _finite_matches(normalize(text or ""))]


def number_tokens(text: str) -> list[str]:
    """The standalone number tokens exactly as written (for rounding comparisons)."""
    return [re.sub(r"\s+", "", m.group()) for m, _ in _finite_matches(normalize(text or ""))]


# ---------------------------------------------------------------------- units

_TEMPERATURE_UNITS = {"c": "C", "degc": "C", "celsius": "C", "k": "K", "kelvin": "K"}
CONDUCTIVITY_FACTORS = {"S/cm": 1.0, "mS/cm": 1e-3, "S/m": 1e-2}  # to S/cm
_CONDUCTIVITY_UNITS = {
    "s/cm": "S/cm", "scm-1": "S/cm", "s.cm-1": "S/cm",
    "ms/cm": "mS/cm", "mscm-1": "mS/cm", "ms.cm-1": "mS/cm",
    "s/m": "S/m", "sm-1": "S/m",
}

# A unit right after a number (or after the last number of a list).
_UNIT_AFTER = re.compile(
    r"\s*(?P<unit>"
    r"(?:°\s*)?C(?![A-Za-z])|degC\b|celsius\b|K(?![A-Za-z])|kelvin\b|"
    r"[mM]?[sS]\s*(?:/|\.|x)?\s*cm(?:\s*\^?\s*-\s*1)?(?![A-Za-z])|[sS]\s*/\s*m\b|"
    r"mol\s*%|at\s*%|wt\s*%|%|m?eV\b|kJ\b|mW|W\b|mV\b|V\b|h\b|hours?\b|min\b|s\b|nm\b|µm|um\b|mm\b|"
    r"MPa\b|Hz\b|g\b|mg\b|mol\b|M\b)",
)
# What may sit between two numbers of one list: "0.02, 0.05 and 0.11 S/cm", "400-600 C".
_LIST_GAP = re.compile(r"\s*(?:,|;|&|/|-|and|or|to|,\s*and|,\s*or)?\s*", re.I)


def _unit_kind(unit: str) -> tuple[str, str]:
    key = unit.strip().lower().replace("°", "").replace(" ", "")
    key = key.replace("x", ".").replace("*", ".").replace("^", "")
    if key in _TEMPERATURE_UNITS:
        return "temperature", _TEMPERATURE_UNITS[key]
    if key in _CONDUCTIVITY_UNITS:
        return "conductivity", _CONDUCTIVITY_UNITS[key]
    return "other", unit.strip()


def quantities(text: str) -> list[tuple[float, str, str | None]]:
    """(value, kind, unit) for each number written in ``text``.

    ``kind`` is "temperature", "conductivity", "other" (mol%, eV, hours, ...), "label"
    (glued to a name, as the 20 in "20SDC") or "bare" (no unit). A unit after the last
    number of a list is given to every number in that list.
    """
    t = normalize(text or "")
    found = _finite_matches(t)
    items: list[list] = []
    for i, (match, value) in enumerate(found):
        next_start = found[i + 1][0].start() if i + 1 < len(found) else len(t)
        gap = t[match.end():next_start]
        unit_match = _UNIT_AFTER.match(gap)
        if unit_match:
            kind, unit = _unit_kind(unit_match.group("unit"))
        elif match.end() < len(t) and t[match.end()].isalpha():
            kind, unit = "label", None
        else:
            kind, unit = "bare", None
        joins_next = not unit_match and i + 1 < len(found) and _LIST_GAP.fullmatch(gap) is not None
        items.append([value, kind, unit, joins_next])
    for i in range(len(items) - 2, -1, -1):  # carry a list's unit back to its earlier items
        if items[i][1] == "bare" and items[i][3] and items[i + 1][1] in ("temperature", "conductivity", "other"):
            items[i][1], items[i][2] = items[i + 1][1], items[i + 1][2]
    return [(value, kind, unit) for value, kind, unit, _ in items]


def parse_quantity(raw: object, kind: str) -> tuple[float, str | None]:
    """Parse one value that may carry a unit.

    ``kind`` is "temperature" (unit "C" or "K") or "conductivity" (unit "S/cm", "mS/cm" or
    "S/m"; the value is returned converted to S/cm). The unit is None when none was
    written. Raises ValueError for anything else, so '5.2 x 10^-3' is never read as 5.2.
    """
    if isinstance(raw, bool):
        raise ValueError(f"not a number: {raw!r}")
    if isinstance(raw, (int, float)):
        if not math.isfinite(float(raw)):
            raise ValueError(f"not a finite number: {raw!r}")
        return float(raw), None
    text = normalize(str(raw)).strip()
    match = NUMBER.match(text)
    if not match:
        raise ValueError(f"not a number: {raw!r}")
    value = to_float(match.group())
    rest = text[match.end():].strip()
    if not rest:
        return value, None
    unit_kind, unit = _unit_kind(rest)
    if unit_kind == kind == "temperature":
        return value, unit
    if unit_kind == kind == "conductivity":
        return value * CONDUCTIVITY_FACTORS[unit], unit
    raise ValueError(f"not a number: {raw!r}")


# ---------------------------------------------------------------------- rounding

def _decimals(token: str) -> int:
    body = token.lstrip("+-")
    return len(body.split(".")[1]) if "." in body and "e" not in body.lower() else 0


def significant_digits(token: str) -> int:
    digits = re.sub(r"[^0-9]", "", token.lstrip("+-").split("e")[0].split("E")[0])
    return len(digits.lstrip("0")) or 1


def reports(token: str, reference: float, allow_truncation: bool = False) -> bool:
    """True when ``token`` is ``reference`` reported to the token's own precision.

    '0.807' and '0.806' both report 0.8065; '5.06' and '5.1' report 5.057; '0.45' does
    not report 0.453, and a one-significant-digit token ('5', '0.5') reports nothing.
    Both the shortest decimal form of the float and its exact binary value are rounded,
    so a tool that printed f"{2.0035:.3f}" ('2.003') is matched too.
    """
    try:
        value = Decimal(token.replace("+", ""))
        reference = float(reference)
    except (InvalidOperation, ValueError, TypeError):
        return False
    if not math.isfinite(reference) or not value.is_finite():
        return False
    refs = {Decimal(repr(reference)), Decimal(reference)}
    if significant_digits(token) < 2:
        return value in refs
    quantum = Decimal(1).scaleb(-_decimals(token))
    modes = (ROUND_HALF_UP, ROUND_HALF_EVEN) + ((ROUND_DOWN,) if allow_truncation else ())
    try:
        return any(value == ref.quantize(quantum, rounding=mode) for ref in refs for mode in modes)
    except InvalidOperation:  # precision overflow for absurd magnitudes
        return False
