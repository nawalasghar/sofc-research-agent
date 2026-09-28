"""Tool registry for the SOFC research agent.

Each tool is a plain Python function that takes JSON-friendly arguments and returns a
JSON-serialisable dict containing ``ok``, a one-line ``summary`` for the model to read, and
the structured detail. Tools report problems in the return value instead of raising, so a
failed call is something the agent can read and recover from.

``TOOL_SCHEMAS`` is the JSON-schema function list in the shape Ollama's ``tools`` parameter
expects; ``call_tool`` dispatches what the model asks for.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

from .arrhenius import arrhenius
from .dataset import query_dataset
from .literature import search_literature
from .materials import materials_project

__all__ = [
    "arrhenius",
    "query_dataset",
    "search_literature",
    "materials_project",
    "TOOLS",
    "TOOL_SCHEMAS",
    "call_tool",
]

TOOLS: dict[str, Callable[..., dict]] = {
    "search_literature": search_literature,
    "query_dataset": query_dataset,
    "arrhenius": arrhenius,
    "materials_project": materials_project,
}

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": (
                "Search the local corpus of SOFC research papers for passages relevant to a "
                "question. Use this for explanations, mechanisms, synthesis routes, or any "
                "'what does the literature say' question. Returns passages with paper title, "
                "page number and a citation string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or keyword phrase to search for.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of passages to return (default 5, max 20).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_dataset",
            "description": (
                "Query the extracted table of measured SOFC electrolyte data (conductivity, "
                "activation energy, peak power, OCV) with one row per reported composition. Use "
                "this for numeric or comparative questions about measured values. Every row "
                "carries the paper_id it came from."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dopant": {
                        "type": "string",
                        "description": (
                            "Ceria dopant element, e.g. 'Sm', 'Ca', 'Al', 'Ba' (also matches "
                            "co-doped entries), or a composition name such as 'LN-SDC'. Omit it "
                            "to include every composition."
                        ),
                    },
                    "carbonate": {
                        "type": "string",
                        "description": (
                            "Omit it to include every composition. Otherwise 'any' for carbonate "
                            "composites, 'none' only for oxides without carbonate, a carbonate "
                            "such as 'Na2CO3' or '(Li/Na)2CO3', or 'binary'/'ternary'."
                        ),
                    },
                    "temperature_C": {
                        "type": "number",
                        "description": "Measurement temperature in Celsius, matched within +/-25 C.",
                    },
                    "temperature_field": {
                        "type": "string",
                        "enum": ["conductivity_temp_C", "pmax_temp_C"],
                        "description": (
                            "What temperature_C refers to: the conductivity measurement "
                            "(default) or the peak power density measurement."
                        ),
                    },
                    "paper_id": {
                        "type": "string",
                        "description": "Restrict to one source paper, e.g. 'ref11_LN-SDC'.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arrhenius",
            "description": (
                "Fit an Arrhenius activation energy (Ea, in eV) to a set of conductivity values "
                "measured at different temperatures. Use this whenever the user supplies "
                "conductivity-versus-temperature data and wants Ea, or asks how activation "
                "energy is obtained from such data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "temperatures": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Measurement temperatures, one per conductivity value.",
                    },
                    "conductivities": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Conductivities in S/cm, in the same order as temperatures.",
                    },
                    "temperature_unit": {
                        "type": "string",
                        "enum": ["C", "K"],
                        "description": "Unit of the temperatures (default 'C').",
                    },
                    "model": {
                        "type": "string",
                        "enum": ["sigma_T", "sigma"],
                        "description": (
                            "'sigma_T' fits ln(sigma*T) vs 1/T (default, usual for oxide-ion "
                            "conductors); 'sigma' fits ln(sigma) vs 1/T."
                        ),
                    },
                },
                "required": ["temperatures", "conductivities"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "materials_project",
            "description": (
                "Look up a chemical formula in the Materials Project database and return its "
                "thermodynamic stability (energy above the convex hull) and DFT band gap. Use "
                "this for questions about whether a phase is stable or whether it is an "
                "insulator. Only ordered phases are indexed, so query parent compounds such as "
                "CeO2, Sm2O3 or Na2CO3 rather than doped compositions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "formula": {
                        "type": "string",
                        "description": "Chemical formula, e.g. 'CeO2', 'Sm2O3', 'Na2CO3'.",
                    },
                },
                "required": ["formula"],
            },
        },
    },
]


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> dict:
    """Dispatch a tool call by name.

    Unknown argument names are dropped rather than raising, because a small model will
    occasionally invent one; the dropped names are reported back in ``ignored_arguments``.
    """
    func = TOOLS.get(name)
    if func is None:
        return {
            "ok": False,
            "tool": name,
            "error": f"unknown tool {name!r}; available tools: {sorted(TOOLS)}",
        }

    arguments = dict(arguments or {})
    signature = inspect.signature(func)
    ignored = sorted(set(arguments) - set(signature.parameters))
    kwargs = {key: val for key, val in arguments.items() if key in signature.parameters}

    try:  # checked before the call, so a TypeError inside a tool is not blamed on the caller
        bound = signature.bind(**kwargs)
    except TypeError as exc:
        return {"ok": False, "tool": name, "error": f"bad arguments for {name}: {exc}"}
    try:
        result = func(*bound.args, **bound.kwargs)
    except Exception as exc:  # a tool bug should not kill the agent loop
        return {"ok": False, "tool": name, "error_code": "tool_exception", "error": f"{type(exc).__name__}: {exc}"}

    if ignored:
        result = dict(result)
        result["ignored_arguments"] = ignored
    return result
