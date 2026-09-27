"""Exercise each tool separately and print what it returns.

    python scripts/test_tools.py                 # all four tools
    python scripts/test_tools.py --tool dataset  # one tool
    python scripts/test_tools.py --raw           # untruncated JSON

This is a demonstration harness, not the test suite. For assertions run:

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sofc_agent.tools import arrhenius, materials_project, query_dataset, search_literature  # noqa: E402
from sofc_agent.tools.arrhenius import KB_EV_PER_K  # noqa: E402

TRUNCATE_AT = 220


def rule(char: str = "=", width: int = 78) -> str:
    return char * width


def banner(text: str) -> None:
    print(f"\n{rule()}\n{text}\n{rule()}")


def case(text: str) -> None:
    print(f"\n--- {text} {rule('-', max(0, 72 - len(text)))}")


def shrink(value, raw: bool):
    """Truncate long strings so a terminal dump stays readable."""
    if raw:
        return value
    if isinstance(value, str) and len(value) > TRUNCATE_AT:
        return value[:TRUNCATE_AT] + f" ... [+{len(value) - TRUNCATE_AT} chars]"
    if isinstance(value, dict):
        return {key: shrink(val, raw) for key, val in value.items()}
    if isinstance(value, list):
        return [shrink(item, raw) for item in value]
    return value


def dump(result: dict, raw: bool = False) -> None:
    print(json.dumps(shrink(result, raw), indent=2, ensure_ascii=False))


def synthetic_series(ea_ev: float, prefactor: float, temps_c: list[float]) -> list[float]:
    """sigma = (A/T) * exp(-Ea / (kB*T)) - used for the round-trip check."""
    out = []
    for t_c in temps_c:
        t_k = t_c + 273.15
        out.append(prefactor / t_k * math.exp(-ea_ev / (KB_EV_PER_K * t_k)))
    return out


def demo_literature(raw: bool) -> None:
    banner("TOOL 1/4  search_literature  (FAISS index of the paper corpus)")

    case("mechanism question, k=3")
    dump(search_literature(
        "Why do ceria-carbonate composite electrolytes conduct so well below 600 C?", k=3
    ), raw)

    case("targeted lookup, k=2")
    dump(search_literature("activation energy of samarium doped ceria electrolyte", k=2), raw)

    case("error handling: empty query")
    dump(search_literature("  "), raw)


def demo_dataset(raw: bool) -> None:
    banner("TOOL 2/4  query_dataset  (extracted measurement table)")

    case("filter by dopant + carbonate + temperature: Sm, any carbonate, 600 C")
    dump(query_dataset(dopant="Sm", carbonate="any", temperature_C=600), raw)

    case("carbonate-free baseline: dopant=Al, carbonate=none")
    dump(query_dataset(dopant="Al", carbonate="none"), raw)

    case("specific secondary phase: carbonate='ternary'")
    dump(query_dataset(carbonate="ternary"), raw)

    case("one paper: paper_id='ref11_LN-SDC' (single vs binary vs ternary carbonates)")
    dump(query_dataset(paper_id="ref11_LN-SDC", limit=8), raw)

    case("no match: dopant='Gd'")
    dump(query_dataset(dopant="Gd"), raw)


def demo_arrhenius(raw: bool) -> None:
    banner("TOOL 3/4  arrhenius  (least-squares fit of Ea in eV)")

    ea_true = 0.45
    temps = [400.0, 450.0, 500.0, 550.0, 600.0]
    sigmas = synthetic_series(ea_true, 1.0e4, temps)

    case(f"round-trip check: data generated with Ea = {ea_true} eV, then fitted")
    result = arrhenius(temperatures=temps, conductivities=sigmas)
    dump(result, raw)
    print(f"\n    recovered {result['activation_energy_eV']} eV vs true {ea_true} eV "
          f"-> error {abs(result['activation_energy_eV'] - ea_true):.2e} eV")

    case("illustrative measured series (rounded values, so the fit shows scatter)")
    dump(arrhenius(
        temperatures=[400, 450, 500, 550, 600],
        conductivities=[0.013, 0.028, 0.055, 0.10, 0.17],
    ), raw)

    case("ln(sigma) convention instead of ln(sigma*T), temperatures in kelvin")
    dump(arrhenius(
        temperatures=[673.15, 723.15, 773.15, 823.15, 873.15],
        conductivities=[0.013, 0.028, 0.055, 0.10, 0.17],
        temperature_unit="K",
        model="sigma",
    ), raw)

    case("error handling: length mismatch")
    dump(arrhenius(temperatures=[500, 550, 600], conductivities=[0.05, 0.1]), raw)

    case("error handling: non-positive conductivity")
    dump(arrhenius(temperatures=[500, 550], conductivities=[0.05, 0.0]), raw)


def demo_materials(raw: bool) -> None:
    banner("TOOL 4/4  materials_project  (free REST API, key from .env)")

    for formula in ("CeO2", "Na2CO3", "Sm2O3"):
        case(f"formula = {formula}")
        dump(materials_project(formula), raw)

    case("doped composition is not indexed: Ce0.8Sm0.2O2")
    dump(materials_project("Ce0.8Sm0.2O2"), raw)

    case("error handling: empty formula")
    dump(materials_project(""), raw)


DEMOS = {
    "literature": demo_literature,
    "dataset": demo_dataset,
    "arrhenius": demo_arrhenius,
    "materials": demo_materials,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run each agent tool and show its output.")
    parser.add_argument("--tool", choices=sorted(DEMOS), action="append",
                        help="run only this tool (repeatable; default all)")
    parser.add_argument("--raw", action="store_true", help="do not truncate long strings")
    args = parser.parse_args()

    for name in (args.tool or list(DEMOS)):
        DEMOS[name](args.raw)

    print(f"\n{rule()}\ndone\n{rule()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
