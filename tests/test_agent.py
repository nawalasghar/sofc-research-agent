"""Tests for the agent loop, driven by a scripted fake model (no Ollama needed).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sofc_agent import agent as agent_mod  # noqa: E402
from sofc_agent.agent import (  # noqa: E402
    NOT_FOUND,
    Agent,
    AgentError,
    OllamaChat,
    _Grounding,
    _Registry,
    _tool_calls_in_text,
    digest,
    resolve_citations,
    says_not_found,
    strip_tool_calls,
    ungrounded_numbers,
)

# ------------------------------------------------------------------ fake backends

PASSAGES = [
    {
        "rank": i + 1,
        "score": score,
        "title": title,
        "page": page,
        "citation": f"[{title}, p.{page}]",
        "file_name": f"{i}.pdf",
        "doi": None,
        "snippet": snippet,
    }
    for i, (score, title, page, snippet) in enumerate([
        (0.75, "Improved ceria-carbonate composite electrolytes", 1,
         "Interfacial conduction along the ceria-carbonate interface raises ionic conductivity."),
        (0.70, "Functional ceria-based nanocomposites for advanced low-temperature SOFCs", 7,
         "Proton and oxide ion co-conduction occurs in the composite."),
        (0.20, "Unrelated review of battery anodes", 3, "Graphite anodes intercalate lithium."),
    ])
]

LIT_RESULT = {"ok": True, "tool": "search_literature", "query": "q", "passages": PASSAGES,
              "summary": "3 passages"}
MP_RESULT = {
    "ok": True, "tool": "materials_project", "found": True, "n_candidates": 1,
    "summary": "CeO2 stable",
    "stability": "on the convex hull - thermodynamically stable at 0 K",
    "band_gap_description": "2.003 eV indirect band gap (DFT/GGA, typically underestimated)",
    "most_stable": {
        "material_id": "mp-aaaabdws", "formula_pretty": "CeO2", "spacegroup_symbol": "Fm-3m",
        "crystal_system": "Cubic", "energy_above_hull_eV_per_atom": 0.0,
        "formation_energy_eV_per_atom": -3.9272, "band_gap_eV": 2.0035,
        "mp_url": "https://materialsproject.org/materials/mp-aaaabdws",
    },
    "polymorphs": [],
}
MP_NOT_FOUND = {"ok": True, "tool": "materials_project", "found": False,
                "summary": "No Materials Project entry matches formula 'X'."}


def fake_dispatch(calls_log: list):
    """Real dataset/arrhenius tools, canned literature and Materials Project results."""
    from sofc_agent.tools import call_tool

    def dispatch(name: str, arguments: dict) -> dict:
        calls_log.append((name, dict(arguments)))
        if name == "search_literature":
            return dict(LIT_RESULT, query=arguments.get("query", ""))
        if name == "materials_project":
            return MP_NOT_FOUND if arguments.get("formula") == "Ce0.8Sm0.2O2" else MP_RESULT
        return call_tool(name, arguments)

    return dispatch


def tool_call(name: str, **arguments) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


class ScriptedChat:
    """Replays a list of replies; records every request it was sent."""

    model = "fake-model"

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.requests: list[tuple[list, list | None]] = []

    def __call__(self, messages, tools=None):
        self.requests.append(([dict(m) for m in messages], tools))
        if not self.replies:
            return {"content": "I am out of script.", "tool_calls": []}
        reply = self.replies.pop(0)
        return {"content": reply.get("content", ""), "tool_calls": reply.get("tool_calls", []),
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def make_agent(replies, max_tool_calls=5):
    log: list = []
    chat = ScriptedChat(replies)
    return Agent(chat=chat, max_tool_calls=max_tool_calls, dispatch=fake_dispatch(log)), chat, log


# ------------------------------------------------------------------------ the loop

class TestAgentLoop(unittest.TestCase):
    def test_dataset_answer_is_cited_and_expanded(self):
        agent, chat, log = make_agent([
            {"tool_calls": [tool_call("query_dataset", dopant="Sm", carbonate="any", temperature_C=600)]},
            {"content": "LN-SDC reaches 0.31 S/cm at 600 C [ref11_LN-SDC]."},
        ])
        result = agent.ask("Which Sm composite conducts best at 600 C?")
        self.assertEqual(result.stop_reason, "answered")
        self.assertTrue(result.cited)
        self.assertFalse(result.not_found)
        self.assertEqual(result.tools_used, ["query_dataset"])
        self.assertIn("[ref11_LN-SDC]", result.answer)
        self.assertIn("Sources:", result.answer)
        self.assertIn("Ali, Raza et al. 2018", result.answer)  # paper citation from the dataset
        self.assertEqual(result.ungrounded_numbers, [])
        self.assertEqual(result.llm_calls, 2)
        self.assertEqual(result.prompt_tokens, 200)

    def test_literature_tags_become_title_and_page(self):
        agent, _, _ = make_agent([
            {"tool_calls": [tool_call("search_literature", query="carbonate composite mechanism")]},
            {"content": "Conduction runs along the ceria-carbonate interface [L1, L2]."},
        ])
        result = agent.ask("Why do carbonate composites conduct well?")
        self.assertTrue(result.cited)
        self.assertIn(
            "[Improved ceria-carbonate composite electrolytes, p.1; "
            "Functional ceria-based nanocomposites for advanced low-temperature SOFCs, p.7]",
            result.answer,
        )
        self.assertEqual([s.tag for s in result.cited_sources], ["L1", "L2"])

    def test_low_relevance_passages_never_reach_the_model(self):
        agent, chat, _ = make_agent([
            {"tool_calls": [tool_call("search_literature", query="anodes")]},
            {"content": "Graphite anodes intercalate lithium [L3]."},
            {"content": "Graphite anodes intercalate lithium [L3]."},  # same after the repair request
        ])
        result = agent.ask("Tell me about graphite anodes")
        tool_messages = [m for m in chat.requests[-1][0] if m["role"] == "tool"]
        self.assertNotIn("battery anodes", tool_messages[0]["content"])
        self.assertIn("low-relevance", tool_messages[0]["content"])
        # L3 was the dropped passage, so the model could only have invented that tag.
        self.assertEqual(result.invalid_citations, ["L3"])
        self.assertIn("[unverified source]", result.answer)
        self.assertTrue(result.repaired)
        self.assertEqual(result.stop_reason, "uncited")
        self.assertFalse(result.cited)

    def test_weak_search_results_are_not_citable(self):
        """Near-domain questions: every passage above the 0.40 floor, none above 0.50."""
        weak = dict(LIT_RESULT, passages=[dict(p, score=s) for p, s in zip(PASSAGES, (0.45, 0.43, 0.42))])
        chat = ScriptedChat([
            {"tool_calls": [tool_call("search_literature", query="melting point of gold nanoparticles")]},
            {"content": "Gold melts at 57 C [L1]."},
        ])
        result = Agent(chat=chat, dispatch=lambda name, args: weak).ask(
            "What melting point do the papers report for gold nanoparticles?"
        )
        self.assertEqual(result.stop_reason, "nothing_to_cite")
        self.assertTrue(result.not_found)
        tool_text = next(m["content"] for m in chat.requests[1][0] if m["role"] == "tool")
        self.assertIn("no passage relevant", tool_text)
        self.assertNotIn("gold", tool_text.lower(), "the model's query is not echoed back")

    def test_tool_call_limit_is_enforced(self):
        endless = [{"tool_calls": [tool_call("query_dataset", dopant=f"Sm{i}")]} for i in range(10)]
        agent, chat, log = make_agent(endless + [{"content": "Done [ref11_LN-SDC]."}], max_tool_calls=5)
        result = agent.ask("Loop forever")
        self.assertEqual(len(log), 5)
        self.assertEqual(len(result.tool_calls), 5)
        tools_sent = [tools for _, tools in chat.requests]
        self.assertTrue(all(t is not None for t in tools_sent[:5]))
        self.assertIsNone(tools_sent[5], "after 5 calls the model must be asked without tools")

    def test_parallel_calls_beyond_the_limit_are_not_executed(self):
        agent, chat, log = make_agent([
            {"tool_calls": [tool_call("query_dataset", dopant=d) for d in ("Sm", "Ca", "Al")]},
            {"content": "LN-SDC conducts best [ref11_LN-SDC]; 30ADC too [ref04_30ADC]."},
        ], max_tool_calls=2)
        result = agent.ask("Three at once")
        self.assertEqual([name for name, _ in log], ["query_dataset", "query_dataset"])
        skipped = [m for m in chat.requests[-1][0] if m["role"] == "tool" and "Not executed" in m["content"]]
        self.assertEqual(len(skipped), 1)
        self.assertIsNone(chat.requests[-1][1], "no tools once the limit is reached")
        self.assertEqual(result.stop_reason, "answered")
        # The Al call never ran, so its paper cannot be cited.
        self.assertEqual(result.invalid_citations, ["ref04_30ADC"])

    def test_no_tool_call_is_nudged_then_refused(self):
        agent, chat, log = make_agent([
            {"content": "Paris is the capital of France."},
            {"content": "Still Paris."},
        ])
        result = agent.ask("What is the capital of France?")
        self.assertTrue(result.nudged)
        self.assertTrue(result.not_found)
        self.assertEqual(result.answer, NOT_FOUND)
        self.assertEqual(result.stop_reason, "no_tool_called")
        self.assertNotIn("Paris", result.answer)
        self.assertEqual(log, [])

    def test_nudge_can_recover(self):
        agent, _, _ = make_agent([
            {"content": "CeO2 is stable."},
            {"tool_calls": [tool_call("materials_project", formula="CeO2")]},
            {"content": "CeO2 is on the hull with a 2.0 eV gap [MP1]."},
        ])
        result = agent.ask("Is CeO2 stable?")
        self.assertTrue(result.nudged)
        self.assertTrue(result.cited)
        self.assertIn("[Materials Project mp-aaaabdws]", result.answer)

    def test_uncited_answer_is_repaired(self):
        agent, chat, _ = make_agent([
            {"tool_calls": [tool_call("materials_project", formula="CeO2")]},
            {"content": "CeO2 is stable with a 2.0 eV band gap."},
            {"content": "CeO2 is stable with a 2.0 eV band gap [MP1]."},
        ])
        result = agent.ask("Is CeO2 stable?")
        self.assertTrue(result.repaired)
        self.assertTrue(result.cited)
        self.assertIn("no source tags", chat.requests[-1][0][-1]["content"])

    def test_answer_still_uncited_after_repair_is_flagged(self):
        agent, _, _ = make_agent([
            {"tool_calls": [tool_call("materials_project", formula="CeO2")]},
            {"content": "CeO2 is stable."},
            {"content": "CeO2 is definitely stable."},
        ])
        result = agent.ask("Is CeO2 stable?")
        self.assertEqual(result.stop_reason, "uncited")
        self.assertFalse(result.cited)
        self.assertFalse(result.not_found)
        self.assertIn("Unverified", result.answer)

    def test_model_saying_not_found_gives_the_standard_sentence(self):
        agent, _, _ = make_agent([
            {"tool_calls": [tool_call("search_literature", query="x")]},
            {"content": "The passages do not contain this information."},
        ])
        result = agent.ask("Something the corpus lacks")
        self.assertTrue(result.not_found)
        self.assertTrue(result.answer.startswith(NOT_FOUND))
        self.assertIn("What the tools returned", result.answer)
        self.assertFalse(result.repaired, "a not-found reply must not be sent for repair")

    def test_empty_tool_results_force_not_found_even_if_the_model_guesses(self):
        agent, chat, _ = make_agent([
            {"tool_calls": [tool_call("query_dataset", dopant="Gd")]},
            {"content": "GDC reaches 0.1 S/cm at 600 C."},
        ])
        result = agent.ask("What conductivity does Gd-doped ceria reach?")
        self.assertTrue(result.not_found)
        self.assertEqual(result.stop_reason, "nothing_to_cite")
        self.assertNotIn("0.1 S/cm", result.answer)
        self.assertEqual(len(chat.requests), 2, "nothing to cite -> no repair round")

    def test_repeated_identical_call_is_served_from_cache(self):
        agent, chat, log = make_agent([
            {"tool_calls": [tool_call("query_dataset", dopant="Al")]},
            {"tool_calls": [tool_call("query_dataset", dopant="Al")]},
            {"content": "30ADC reaches 0.18 S/cm at 500 C [ref04_30ADC]."},
        ])
        result = agent.ask("Al-doped ceria?")
        self.assertEqual(len(log), 1, "the tool itself runs once")
        self.assertEqual(len(result.tool_calls), 2)
        self.assertTrue(result.tool_calls[1].repeated)
        self.assertIn("already made this exact call", chat.requests[2][0][-1]["content"])

    def test_tool_call_written_as_text_is_executed(self):
        agent, _, log = make_agent([
            {"content": '<tool_call>{"name": "query_dataset", "arguments": {"dopant": "Al"}}</tool_call>'},
            {"content": "30ADC reaches 0.18 S/cm [ref04_30ADC]."},
        ])
        result = agent.ask("Al-doped ceria?")
        self.assertEqual(log, [("query_dataset", {"dopant": "Al"})])
        self.assertTrue(result.cited)

    def test_arguments_sent_as_a_json_string_are_parsed(self):
        agent, _, log = make_agent([
            {"tool_calls": [{"function": {"name": "query_dataset", "arguments": '{"dopant": "Ca"}'}}]},
            {"content": "NK-CDC [ref10_NK-CDC]."},
        ])
        agent.ask("Ca?")
        self.assertEqual(log[0], ("query_dataset", {"dopant": "Ca"}))

    def test_unknown_tool_is_reported_to_the_model(self):
        agent, chat, log = make_agent([
            {"tool_calls": [tool_call("predict_future", year=2030)]},
            {"content": "I cannot answer."},
        ])
        result = agent.ask("Predict")
        self.assertEqual(log, [])
        self.assertIn("unknown tool", chat.requests[1][0][-1]["content"])
        self.assertTrue(result.not_found)

    def test_arguments_outside_the_schema_are_stripped(self):
        agent, chat, log = make_agent([
            {"tool_calls": [tool_call("query_dataset", dopant="Sm", limit=1, full_rows=True)]},
            {"content": "LN-SDC [ref11_LN-SDC]."},
        ])
        result = agent.ask("Sm?")
        self.assertEqual(log[0], ("query_dataset", {"dopant": "Sm"}))
        self.assertEqual(result.tool_calls[0].ignored_arguments, ["full_rows", "limit"])
        self.assertIn("ignored unknown arguments", chat.requests[1][0][-1]["content"])

    def test_invented_arrhenius_data_is_refused(self):
        agent, chat, log = make_agent([
            {"tool_calls": [tool_call("arrhenius", temperatures=[300, 400, 500],
                                      conductivities=[0.00012, 0.00014, 0.00016])]},
            {"content": "Ea is 0.06 eV [A1]."},
        ])
        result = agent.ask("What is the activation energy of N-SDC?")
        self.assertEqual(log, [], "the fit must not run on invented data")
        self.assertIn("never invent measurements", chat.requests[1][0][-1]["content"])
        self.assertTrue(result.not_found)

    def test_arrhenius_on_data_from_the_question_runs(self):
        agent, _, log = make_agent([
            {"tool_calls": [tool_call("arrhenius", temperatures=[400, 450, 500, 550, 600],
                                      conductivities=[0.013, 0.028, 0.055, 0.10, 0.17])]},
            {"content": "Ea = 0.717 eV [A1]."},
        ])
        result = agent.ask(
            "Find Ea for 0.013, 0.028, 0.055, 0.10 and 0.17 S/cm at 400, 450, 500, 550 and 600 C."
        )
        self.assertEqual(len(log), 1)
        self.assertTrue(result.cited)
        self.assertIn("[Arrhenius fit]", result.answer)
        self.assertEqual(result.ungrounded_numbers, [])

    def test_hallucinated_number_is_reported(self):
        agent, _, _ = make_agent([
            {"tool_calls": [tool_call("materials_project", formula="CeO2")]},
            {"content": "CeO2 has a band gap of 3.2 eV [MP1]."},
        ])
        result = agent.ask("Band gap of CeO2?")
        self.assertIn("3.2", result.ungrounded_numbers)

    def test_backend_failure_propagates_as_agent_error(self):
        def broken(messages, tools=None):
            raise AgentError("cannot reach Ollama")

        with self.assertRaises(AgentError):
            Agent(chat=broken).ask("anything")


class TestReviewRegressions(unittest.TestCase):
    """Each case reproduces a defect found in review; the id in the docstring names it."""

    def test_refused_fit_cannot_be_laundered_by_a_retry(self):
        """arr-1: the refusal used to list the invented values, which then 'grounded' a
        retry that differed only by a default argument."""
        series = dict(temperatures=[300, 400, 500], conductivities=[0.00012, 0.00014, 0.00016])
        agent, chat, log = make_agent([
            {"tool_calls": [tool_call("arrhenius", **series)]},
            {"tool_calls": [tool_call("arrhenius", temperature_unit="C", **series)]},
            {"tool_calls": [tool_call("arrhenius", model="sigma_T", temperature_unit="C", **series)]},
            {"content": "The activation energy of N-SDC is 0.06 eV [A1]."},
        ])
        result = agent.ask("What is the activation energy of N-SDC?")
        self.assertEqual(log, [], "the fit must never run on the invented series")
        self.assertTrue(result.not_found)
        self.assertFalse(result.cited)
        refusal = next(m["content"] for m in chat.requests[1][0] if m["role"] == "tool")
        self.assertNotIn("0.00012", refusal, "the refusal must not echo invented values")

    def test_echoed_search_query_does_not_ground_a_fit(self):
        """arr-1 (verifier): numbers in a model's own search query came back in the
        'no relevant passage' digest and grounded the fit."""
        agent, _, log = make_agent([
            {"tool_calls": [tool_call("search_literature", query="N-SDC 300 400 500 C 0.00012 0.00014 S/cm")]},
            {"tool_calls": [tool_call("arrhenius", temperatures=[300, 400], conductivities=[0.00012, 0.00014])]},
            {"content": "Ea is 0.06 eV [A1]."},
        ])
        agent.ask("What is the activation energy of N-SDC?")
        self.assertEqual([name for name, _ in log], ["search_literature"])

    def test_dataset_values_times_1000_do_not_ground_temperatures(self):
        """guard-2: Ea 0.4 and 0.45 eV used to 'ground' 400 and 450 C."""
        agent, _, log = make_agent([
            {"tool_calls": [tool_call("query_dataset", dopant="Sm")]},
            {"tool_calls": [tool_call("arrhenius", temperatures=[400, 450, 500, 550, 600],
                                      conductivities=[0.02, 0.052, 0.1, 0.2, 0.31])]},
            {"content": "Ea = 0.76 eV [A1]."},
        ])
        agent.ask("What activation energy do Sm-doped ceria carbonate composites have? Use an Arrhenius fit.")
        self.assertEqual([name for name, _ in log], ["query_dataset"])

    def test_stale_pre_nudge_answer_is_never_returned(self):
        """loop-stale-1: a memory answer rejected by the nudge came back as 'cited' when
        the loop ended on the model-call limit."""
        stuck = '<tool_call>{"name": "query_dataset", "arguments": {"dopant": "Sm"}}</tool_call>'
        agent, _, _ = make_agent(
            [{"content": "LN-SDC reaches 1.5 S/cm at 600 C [ref11_LN-SDC]."}]
            + [{"tool_calls": [tool_call("query_dataset", dopant=d)]} for d in ("Sm", "Gd", "Y", "La", "Nd")]
            + [{"content": stuck}] * 5
        )
        result = agent.ask("Which composite conducts best?")
        self.assertTrue(result.not_found)
        self.assertFalse(result.cited)
        self.assertNotIn("1.5 S/cm", result.answer)

    def test_answer_sent_with_a_text_tool_call_after_the_limit_is_kept(self):
        """loop-postlimit-2: the answer part was thrown away with the extra call."""
        agent, _, log = make_agent([
            {"tool_calls": [tool_call("query_dataset", dopant=d) for d in ("Sm", "Ca", "Al", "Gd", "Y")]},
            {"content": 'LN-SDC reaches 0.31 S/cm at 600 C [ref11_LN-SDC].\n'
                        '<tool_call>{"name": "query_dataset", "arguments": {"dopant": "La"}}</tool_call>'},
        ])
        result = agent.ask("Which composite conducts best?")
        self.assertEqual(len(log), 5)
        self.assertEqual(result.stop_reason, "answered")
        self.assertTrue(result.cited)
        self.assertNotIn("tool_call", result.answer)

    def test_a_second_post_limit_tool_request_stops_the_loop(self):
        stuck = {"tool_calls": [tool_call("query_dataset", dopant="Sm")]}
        agent, chat, _ = make_agent([
            {"tool_calls": [tool_call("query_dataset", dopant=d) for d in ("Sm", "Ca", "Al", "Gd", "Y")]},
            stuck, stuck, stuck,
        ])
        result = agent.ask("loop")
        self.assertEqual(result.stop_reason, "tool_limit")
        self.assertTrue(result.not_found)
        self.assertEqual(result.llm_calls, 2, "LIMIT_REACHED is sent at most once")

    def test_near_miss_phrases_are_not_declines(self):
        """nf-substring-3 / nf-1: 'there is now' and 'there are notable' matched as declines."""
        agent, _, _ = make_agent([
            {"tool_calls": [tool_call("search_literature", query="mechanism")]},
            {"content": "There is now broad agreement that conduction runs along the interface."},
            {"content": "There is now broad agreement that conduction runs along the interface [L1]."},
        ])
        result = agent.ask("What is the mechanism?")
        self.assertTrue(result.repaired)
        self.assertTrue(result.cited)
        self.assertFalse(result.not_found)
        for text in ("There are notable differences.", "no data loss after 100 h at 0.9 V",
                     "NK-CDC reaches 567 mW/cm2; there is no carbonate loss reported."):
            self.assertFalse(says_not_found(text), text)

    def test_not_found_sentence_with_a_tag_is_a_decline(self):
        """nf-cited-4: '<not-found sentence> [L1]' was scored as a cited answer."""
        agent, _, _ = make_agent([
            {"tool_calls": [tool_call("search_literature", query="gold")]},
            {"content": NOT_FOUND + " [L1]"},
        ])
        result = agent.ask("What melting point do the papers report for gold nanoparticles?")
        self.assertTrue(result.not_found)
        self.assertFalse(result.cited)

    def test_list_valued_tool_name_does_not_crash(self):
        """parse-crash-5."""
        agent, _, log = make_agent([
            {"content": '{"name": ["query_dataset", "search_literature"], "arguments": {"dopant": "Sm"}}'},
            {"content": '{"name": {"x": 1}}'},
        ])
        result = agent.ask("Sm?")
        self.assertEqual(log, [])
        self.assertTrue(result.not_found)

    def test_empty_final_answer_is_not_found(self):
        agent, _, _ = make_agent([
            {"tool_calls": [tool_call("query_dataset", dopant="Sm")]},
            {"content": ""},
            {"content": ""},
        ])
        result = agent.ask("Sm?")
        self.assertEqual(result.stop_reason, "empty_answer")
        self.assertTrue(result.not_found)
        self.assertEqual(result.answer.split("\n")[0], NOT_FOUND)

    def test_infrastructure_error_code_is_recorded(self):
        def dispatch(name, arguments):
            return {"ok": False, "tool": name, "error_code": "rate_limited", "error": "HTTP 429"}

        chat = ScriptedChat([{"tool_calls": [tool_call("materials_project", formula="CeO2")]},
                             {"content": "I cannot answer."}])
        result = Agent(chat=chat, dispatch=dispatch).ask("CeO2?")
        self.assertEqual(result.tool_calls[0].error_code, "rate_limited")


class TestRoundTwoRegressions(unittest.TestCase):
    """Defects from the second review round (ids in the docstrings)."""

    def run_fit(self, question, temperatures, conductivities):
        agent, _, log = make_agent([
            {"tool_calls": [tool_call("arrhenius", temperatures=temperatures, conductivities=conductivities)]},
            {"content": "The activation energy is 0.82 eV [A1]."},
        ])
        return agent.ask(question), log

    def test_numbers_with_other_units_do_not_ground_conductivities(self):
        """F1: '20 mol%' grounded 0.02 and 0.2; '20SDC' likewise."""
        for question, sigmas in [
            ("What is the activation energy of 20 mol% Sm-doped ceria between 500 and 700 C?", [0.02, 0.2]),
            ("What is the activation energy of 10% Gd-doped ceria from 500 to 700 C?", [0.01, 0.1]),
            ("What is the activation energy of 20SDC between 400 C and 600 C?", [0.02, 0.2]),
        ]:
            with self.subTest(question):
                result, log = self.run_fit(question, [500, 700] if "700" in question else [400, 600], sigmas)
                self.assertEqual(log, [])
                self.assertFalse(result.cited)

    def test_a_unit_written_once_after_a_list_grounds_the_whole_list(self):
        result, log = self.run_fit(
            "A composite has 0.02, 0.05 and 0.11 S/cm at 773, 823 and 873 K. What is Ea?",
            [773, 823, 873], [0.02, 0.05, 0.11],
        )
        self.assertEqual(len(log), 1)
        self.assertTrue(result.cited)

    def test_partial_answer_after_the_not_found_sentence_is_kept(self):
        """F3: a cited value after a leading not-found clause was thrown away."""
        agent, _, _ = make_agent([
            {"tool_calls": [tool_call("query_dataset", paper_id="ref11")]},
            {"content": NOT_FOUND[:-1] + " for the stability part. N-SDC reaches 0.052 S/cm at 600 C [ref11_LN-SDC]."},
        ])
        result = agent.ask("What conductivity does N-SDC reach, and is Na2CO3 stable?")
        self.assertEqual(result.stop_reason, "answered")
        self.assertTrue(result.cited)
        self.assertIn("0.052", result.answer_body)

    def test_scientific_notation_and_echoed_queries_do_not_hide_invented_numbers(self):
        """F4."""
        self.assertEqual(ungrounded_numbers("GDC reaches 7.7 x 10^-2 S/cm", "LN-SDC 0.31 S/cm at 600 C"),
                         ["7.7x10^-2"])
        self.assertEqual(ungrounded_numbers("reaches 3.1 x 10^-1 S/cm", "LN-SDC 0.31 S/cm"), [])
        agent, _, _ = make_agent([
            {"tool_calls": [tool_call("search_literature", query="composite conductivity 0.47 S/cm at 525 C")]},
            {"content": "The composite reaches 0.47 S/cm at 525 C [L1]."},
        ])
        result = agent.ask("How conductive are carbonate composites?")
        self.assertEqual(result.ungrounded_numbers, ["0.47", "525"])

    def test_negation_inside_a_real_answer_is_not_a_decline(self):
        """F5: 'there is no electronic short circuit' declined a correct answer."""
        agent, _, _ = make_agent([
            {"tool_calls": [tool_call("search_literature", query="carbonate shell")]},
            {"content": "The amorphous carbonate shell coats the particles and blocks electron transfer, "
                        "so there is no electronic short circuit through the electrolyte."},
            {"content": "The shell blocks electron transfer between ceria particles [L1]."},
        ])
        result = agent.ask("What role does the carbonate shell play?")
        self.assertTrue(result.repaired)
        self.assertTrue(result.cited)
        self.assertFalse(result.not_found)
        self.assertFalse(says_not_found("The most stable Sm2O3 structure is Ia-3; there is no lower-energy polymorph."))

    def test_restating_the_question_is_still_a_decline(self):
        question = "What is the activation energy of an electrolyte whose conductivity is 0.05 S/cm at 600 C?"
        reply = "The activation energy cannot be determined from a single value of 0.05 S/cm at 600 C."
        self.assertTrue(says_not_found(reply, known=question))
        self.assertFalse(says_not_found(reply.replace("0.05 S/cm", "0.07 S/cm"), known=question))

    def test_other_tool_call_shapes_are_parsed(self):
        """F6: the nested 'function' form and single-quoted dicts."""
        nested = '<tool_call>{"type": "function", "function": {"name": "query_dataset", "arguments": {"dopant": "Al"}}}</tool_call>'
        quoted = "<tool_call>{'name': 'query_dataset', 'arguments': {'dopant': 'Al'}}</tool_call>"
        for text in (nested, quoted):
            with self.subTest(text[:40]):
                agent, _, log = make_agent([{"content": text}, {"content": "30ADC reaches 0.18 S/cm [ref04_30ADC]."}])
                result = agent.ask("Al?")
                self.assertEqual(log, [("query_dataset", {"dopant": "Al"})])
                self.assertTrue(result.cited)
                self.assertNotIn("tool_call", result.answer)

    def test_an_unreadable_tool_call_is_never_an_answer(self):
        garbled = '<tool_call>{"name": "query_dataset", "arguments": {"paper_id": "ref11_LN-SDC",}</tool_call>'
        agent, chat, _ = make_agent([
            {"tool_calls": [tool_call("query_dataset", dopant="Sm")]},
            {"content": garbled},
            {"content": garbled},
        ])
        result = agent.ask("Sm?")
        self.assertEqual(result.stop_reason, "malformed_tool_call")
        self.assertTrue(result.not_found)
        self.assertFalse(result.cited)
        self.assertIn("could not be read", chat.requests[2][0][-1]["content"])

    def test_argument_shapes_are_fitted_to_the_schema(self):
        """eval/F6: ['CeO2'] and schema echoes crashed materials_project (tool_exception)."""
        for formula in (["CeO2"], {"type": "string", "value": "CeO2"}):
            with self.subTest(formula=formula):
                agent, _, log = make_agent([
                    {"tool_calls": [tool_call("materials_project", formula=formula)]},
                    {"content": "CeO2 is stable [MP1]."},
                ])
                agent.ask("CeO2?")
                self.assertEqual(log, [("materials_project", {"formula": "CeO2"})])
        agent, chat, log = make_agent([
            {"tool_calls": [tool_call("materials_project", formula={"nested": True})]},
            {"content": "I cannot answer."},
        ])
        result = agent.ask("CeO2?")
        self.assertEqual(log, [])
        self.assertIsNone(result.tool_calls[0].error_code, "a bad argument is the model's, not infrastructure")
        self.assertIn("must be a string", chat.requests[1][0][-1]["content"])

    def test_garbled_pdf_numbers_do_not_crash(self):
        """parsing/F1: '600e800' parsed as infinity and crashed the number check."""
        garbled = dict(LIT_RESULT, passages=[dict(PASSAGES[0], snippet="SOFCs run at (600e800 /C14 C) and 22e291.")])
        chat = ScriptedChat([
            {"tool_calls": [tool_call("search_literature", query="operating temperature")]},
            {"content": "Conventional SOFCs run at 600-800 C [L1]."},
        ])
        result = Agent(chat=chat, dispatch=lambda name, args: garbled).ask("What temperature do SOFCs run at?")
        self.assertEqual(result.stop_reason, "answered")


class TestRoundThreeRegressions(unittest.TestCase):
    def fit(self, question, **arguments):
        agent, _, log = make_agent([
            {"tool_calls": [tool_call("arrhenius", **arguments)]},
            {"content": "Ea is given by the fit [A1]."},
        ])
        return agent.ask(question), log

    def test_s_cm_minus_one_notation_is_a_conductivity_unit(self):
        """F1: 'S cm-1' is how papers write it; every fit on such data was refused."""
        for unit in ("S cm-1", "S cm^-1", "S cm⁻¹", "S·cm⁻¹", "Scm-1"):
            question = (f"Calculate the activation energy from 0.013 {unit} at 400 C, 0.028 {unit} at 450 C "
                        f"and 0.17 {unit} at 600 C.")
            with self.subTest(unit):
                result, log = self.fit(question, temperatures=[400, 450, 600], conductivities=[0.013, 0.028, 0.17])
                self.assertEqual(len(log), 1)
                self.assertTrue(result.cited)
        self.assertEqual(ungrounded_numbers("LN-SDC reaches 0.31 S cm⁻¹ [ref11_LN-SDC]", "0.31 S/cm"), [])

    def test_kelvin_written_temperatures_are_fitted_as_kelvin(self):
        """F3: q12 without temperature_unit gave a cited 1.857 eV instead of 1.062 eV."""
        question = ("A composite electrolyte has conductivities of 0.02, 0.05 and 0.11 S/cm at 773 K, "
                    "823 K and 873 K. What is its activation energy in eV?")
        _, log = self.fit(question, temperatures=[773, 823, 873], conductivities=[0.02, 0.05, 0.11])
        self.assertEqual(log[0][1]["temperature_unit"], "K")
        agent, chat, _ = make_agent([
            {"tool_calls": [tool_call("arrhenius", temperatures=[773, 823, 873], conductivities=[0.02, 0.05, 0.11])]},
            {"content": "Ea = 1.0623 eV [A1]."},
        ])
        result = agent.ask(question)
        self.assertIn("fitted as kelvin", chat.requests[1][0][-1]["content"])
        self.assertIn("1.0623", result.tool_calls[0].summary)
        # a model that converted to Celsius itself, and Celsius questions, are left alone
        _, log = self.fit(question, temperatures=[499.85, 549.85, 599.85], conductivities=[0.02, 0.05, 0.11])
        self.assertNotIn("temperature_unit", log[0][1])
        _, log = self.fit("Fit 0.013, 0.055 and 0.17 S/cm at 400, 500 and 600 C.",
                          temperatures=[400, 500, 600], conductivities=[0.013, 0.055, 0.17])
        self.assertNotIn("temperature_unit", log[0][1])

    def test_null_and_empty_optional_arguments_mean_not_given(self):
        agent, _, log = make_agent([
            {"tool_calls": [tool_call("query_dataset", dopant="Al", temperature_field=None, paper_id="")]},
            {"content": "30ADC reaches 0.18 S/cm [ref04_30ADC]."},
        ])
        result = agent.ask("Al?")
        self.assertEqual(log, [("query_dataset", {"dopant": "Al"})])
        self.assertTrue(result.cited)

    def test_more_decline_wordings(self):
        """eval-R3-3: declines worded like the arrhenius error."""
        question = "What is the activation energy of an electrolyte whose conductivity is 0.05 S/cm at 600 C?"
        for reply in (
            "The Arrhenius fit needs at least 2 points to fit a slope, but only one point was given.",
            "At least 2 temperatures are needed to calculate the activation energy.",
            "One conductivity value is not enough to calculate the activation energy.",
            "The activation energy can't be calculated from 0.05 S/cm at 600 C alone.",
            "Two or more conductivity measurements are required.",
        ):
            with self.subTest(reply):
                self.assertTrue(says_not_found(reply, known=question))
        self.assertFalse(says_not_found("At least 2 points are needed, and the activation energy is about 0.7 eV.",
                                        known=question))


# ------------------------------------------------------------------ pure helpers

class TestCitationResolution(unittest.TestCase):
    def setUp(self):
        self.reg = _Registry()
        digest("search_literature", LIT_RESULT, self.reg)
        digest("materials_project", MP_RESULT, self.reg)
        from sofc_agent.tools import query_dataset
        digest("query_dataset", query_dataset(paper_id="ref11"), self.reg)

    def test_forms_that_resolve(self):
        for text, tag in [
            ("x [L1].", "L1"),
            ("x (L2).", "L2"),
            ("x L1 and more.", "L1"),
            ("x [MP1].", "MP1"),
            ("x [Materials Project].", "MP1"),
            ("x [ref11_LN-SDC].", "ref11_LN-SDC"),
            ("x [ref11].", "ref11_LN-SDC"),
            ("x per ref11_LN-SDC.", "ref11_LN-SDC"),
            ("x [Improved ceria-carbonate composite electrolytes, p.1].", "L1"),
            ("x [Improved ceria-carbonate composite, page 1].", "L1"),
        ]:
            with self.subTest(text):
                result = resolve_citations(text, self.reg)
                self.assertEqual([s.tag for s in result.valid], [tag])
                self.assertEqual(result.invalid, [])

    def test_wrong_page_does_not_resolve(self):
        result = resolve_citations("x [Improved ceria-carbonate composite electrolytes, p.9].", self.reg)
        self.assertEqual(result.valid, [])

    def test_invented_tags_are_flagged_and_replaced(self):
        result = resolve_citations("x [L9] and [ref99_FAKE] and [MP4].", self.reg)
        self.assertEqual(result.valid, [])
        self.assertEqual(result.invalid, ["L9", "ref99_FAKE", "MP4"])
        self.assertEqual(result.text.count("[unverified source]"), 3)

    def test_mixed_bracket_keeps_the_valid_part(self):
        result = resolve_citations("x [L1, L9].", self.reg)
        self.assertEqual([s.tag for s in result.valid], ["L1"])
        self.assertEqual(result.invalid, ["L9"])

    def test_ordinary_brackets_are_left_alone(self):
        text = "the (Li/Na)2CO3 phase [see Figure 3] melts"
        result = resolve_citations(text, self.reg)
        self.assertEqual(result.text, text)
        self.assertEqual(result.invalid, [])

    def test_a_paper_id_first_in_a_bracket_keeps_the_other_sources(self):
        """cite-1: '[ref11_LN-SDC, MP1]' dropped MP1 and hid invented tags."""
        both = resolve_citations("x [ref11_LN-SDC, MP1].", self.reg)
        self.assertEqual({s.tag for s in both.valid}, {"ref11_LN-SDC", "MP1"})
        fake = resolve_citations("x [ref11_LN-SDC, ref99_FAKE] [ref11_LN-SDC, L7].", self.reg)
        self.assertEqual(fake.invalid, ["ref99_FAKE", "L7"])
        mixed = resolve_citations(
            "x [MP1; Improved ceria-carbonate composite electrolytes, p.1].", self.reg
        )
        self.assertEqual({s.tag for s in mixed.valid}, {"MP1", "L1"})

    def test_materials_project_ids_and_expanded_forms_resolve(self):
        """cite-2."""
        for text in ("CeO2 is stable [mp-aaaabdws].", "CeO2 is stable (mp-aaaabdws).",
                     "CeO2 is stable [Materials Project mp-aaaabdws].", "CeO2 (mp-aaaabdws) is stable."):
            with self.subTest(text):
                self.assertEqual([s.tag for s in resolve_citations(text, self.reg).valid], ["MP1"])

    def test_prose_is_not_read_as_a_citation(self):
        reg = _Registry()
        from sofc_agent.tools import arrhenius

        digest("arrhenius", arrhenius([500, 600], [0.05, 0.17]), reg)
        result = resolve_citations("The A1 site of the perovskite is occupied.", reg)
        self.assertEqual(result.valid, [])
        self.assertEqual(result.text, "The A1 site of the perovskite is occupied.")

    def test_tags_joined_by_words_resolve_and_invented_ones_are_flagged(self):
        """round 2 F2."""
        for text, tags in [("x [L1 and L2].", {"L1", "L2"}), ("x [Source: L1].", {"L1"}),
                           ("x [L1/L2].", {"L1", "L2"}), ("x [L1 & L2].", {"L1", "L2"}),
                           ("x [see L1].", {"L1"}), ("x [MP1 (CeO2)].", {"MP1"}),
                           ("x [L1: Improved ceria-carbonate composite electrolytes, p.1].", {"L1"})]:
            with self.subTest(text):
                result = resolve_citations(text, self.reg)
                self.assertEqual({s.tag for s in result.valid}, tags)
                self.assertEqual(result.invalid, [])
        mixed = resolve_citations("LN-SDC reaches 0.31 S/cm [ref11_LN-SDC and ref99_FAKE].", self.reg)
        self.assertEqual(mixed.invalid, ["ref99_FAKE"])
        self.assertNotIn("ref99_FAKE", mixed.text)
        fake = resolve_citations("x [A Paper Nobody Wrote, p.3].", self.reg)
        self.assertEqual(fake.invalid, ["A Paper Nobody Wrote, p.3"])

    def test_words_inside_a_citation_bracket_are_kept(self):
        result = resolve_citations("x [ref11_LN-SDC, at 600 C].", self.reg)
        self.assertEqual(result.text, "x [ref11_LN-SDC] (at 600 C).")
        self.assertEqual([s.tag for s in result.valid], ["ref11_LN-SDC"])


class TestDigestInvariants(unittest.TestCase):
    def test_every_tag_shown_to_the_model_is_citable(self):
        """A tag printed in a digest but missing from the registry would make the agent
        reject a correct citation (the bug: the best-row statistic named a paper whose
        rows had been cut by the prompt cap)."""
        from sofc_agent.tools import arrhenius, query_dataset

        results = [
            ("query_dataset", query_dataset()),
            ("query_dataset", query_dataset(dopant="Sm")),
            ("query_dataset", query_dataset(carbonate="any")),
            ("query_dataset", query_dataset(paper_id="ref05")),
            ("query_dataset", query_dataset(paper_id="ref09")),
            ("materials_project", MP_RESULT),
            ("arrhenius", arrhenius([500, 600], [0.05, 0.17])),
            ("search_literature", LIT_RESULT),
        ]
        for name, result in results:
            reg = _Registry()
            text, _ = digest(name, result, reg)
            shown = set(re.findall(r"\[((?:L|MP|A)\d+|ref\d+[\w.+-]*)\]", text))
            with self.subTest(name=name, filters=result.get("filters_applied")):
                self.assertTrue(shown)
                for tag in shown:
                    self.assertIsNotNone(reg.get(tag), f"{tag} is shown but not citable")

    def test_measured_rows_are_shown_before_empty_ones(self):
        from sofc_agent.tools import query_dataset

        reg = _Registry()
        text, _ = digest("query_dataset", query_dataset(dopant="Sm"), reg)
        self.assertIn("LN-SDC", text)  # ref11 sits after 12 unmeasured-or-other rows in the CSV
        self.assertIn("report no measured values", text)


class TestHelpers(unittest.TestCase):
    def test_not_found_detection(self):
        self.assertTrue(says_not_found(NOT_FOUND))
        self.assertTrue(says_not_found("The dataset does not contain Gd-doped ceria."))
        self.assertFalse(says_not_found("LN-SDC reaches 0.31 S/cm."))

    def test_ungrounded_numbers_allow_rounding_and_unit_changes(self):
        evidence = "Ea = 0.7170 eV (69.2 kJ/mol) over 400-600 C; hull 0.119 eV/atom"
        self.assertEqual(ungrounded_numbers("Ea is 0.72 eV or 69 kJ/mol from 400 to 600 C", evidence), [])
        self.assertEqual(ungrounded_numbers("119 meV/atom, i.e. 673 K to 873 K", evidence), [])
        self.assertEqual(ungrounded_numbers("Ea is 0.55 eV", evidence), ["0.55"])
        self.assertEqual(ungrounded_numbers("CeO2 and Na2CO3 [Title, p.17]", evidence), [])

    def test_ungrounded_numbers_handle_halves_signs_and_notation(self):
        """num-1 and the dropped rounding note: 0.7150 -> 0.72 is a correct rounding."""
        self.assertEqual(ungrounded_numbers("Ea 0.72 eV", "Ea = 0.7150 eV"), [])
        self.assertEqual(ungrounded_numbers("from 0.013 S/cm to 0.17 S/cm", "1.3e-2 and 1.7E-1 S/cm"), [])
        self.assertEqual(ungrounded_numbers("-3.93 eV/atom between 400 and 600 C",
                                            "over 400-600 C; formation energy -3.9272 eV/atom"), [])

    def test_grounding_by_role_with_unit_changes(self):
        grounding = _Grounding("0.013 S/cm at 400 C, 1.7 x 10^-1 S/cm at 600 C, 55 mS/cm at 773 K")
        self.assertEqual(grounding.invented([400, 600, 773, 673.15], [0.013, 0.17, 0.055, 13]), [])
        self.assertEqual(grounding.invented([450], [0.02]), [450, 0.02])

    def test_grounding_reads_typeset_scientific_notation(self):
        question = "Fit Ea: 5.2×10⁻³, 1.1×10⁻² S/cm at 400 and 450 °C."
        self.assertEqual(_Grounding(question).invented([400, 450], [0.0052, 0.011]), [])

    def test_grounding_ignores_failed_results_and_other_roles(self):
        grounding = _Grounding("What is the activation energy of Sm-doped ceria?")
        grounding.add("arrhenius", {"ok": False, "error": "refused: 300, 400, 0.00012"})
        self.assertEqual(grounding.invented([300, 400], [0.00012]), [300, 400, 0.00012])
        from sofc_agent.tools import query_dataset

        grounding.add("query_dataset", query_dataset(dopant="Sm"))
        # Ea values (0.4, 0.45 eV) times 1000 used to "ground" 400 and 450 C.
        self.assertEqual(grounding.invented([400, 450], []), [400, 450])
        # dataset values are S/cm already: 0.12 must not ground 0.00012
        self.assertEqual(grounding.invented([], [0.00012, 0.00031]), [0.00012, 0.00031])
        # real dataset values (CSDC was measured at 300 C) are grounded, in C or K
        self.assertEqual(grounding.invented([300, 600, 873.15], [0.31, 0.12]), [])

    def test_text_tool_call_parser_ignores_unrelated_json(self):
        text = 'Here {"a": 1} and {"name": "arrhenius", "arguments": {"temperatures": [1, 2]}}'
        self.assertEqual(
            _tool_calls_in_text(text, {"arrhenius"}),
            [("arrhenius", {"temperatures": [1, 2]})],
        )

    def test_strip_tool_calls_keeps_the_prose(self):
        text = 'Answer [L1].\n<tool_call>{"name": "arrhenius", "arguments": {}}</tool_call>'
        self.assertEqual(strip_tool_calls(text, {"arrhenius"}), "Answer [L1].")


class TestOllamaChat(unittest.TestCase):
    def test_request_shape_and_reply_parsing(self):
        reply = mock.Mock(status_code=200)
        reply.json.return_value = {
            "message": {"role": "assistant", "content": "", "tool_calls": [tool_call("arrhenius")]},
            "prompt_eval_count": 1200, "eval_count": 40,
        }
        chat = OllamaChat(model="m", url="http://host:1", num_ctx=4096, seed=7)
        with mock.patch("requests.post", return_value=reply) as post:
            out = chat([{"role": "user", "content": "hi"}], tools=[{"x": 1}])
        url, kwargs = post.call_args[0][0], post.call_args[1]
        self.assertEqual(url, "http://host:1/api/chat")
        self.assertEqual(kwargs["json"]["model"], "m")
        self.assertFalse(kwargs["json"]["stream"])
        self.assertEqual(kwargs["json"]["tools"], [{"x": 1}])
        self.assertEqual(kwargs["json"]["options"]["num_ctx"], 4096)
        self.assertEqual(kwargs["json"]["options"]["seed"], 7)
        self.assertEqual(out["usage"], {"prompt_tokens": 1200, "completion_tokens": 40})
        self.assertEqual(out["tool_calls"][0]["function"]["name"], "arrhenius")

    def test_no_tools_key_when_tools_are_withheld(self):
        reply = mock.Mock(status_code=200)
        reply.json.return_value = {"message": {"content": "ok"}}
        with mock.patch("requests.post", return_value=reply) as post:
            OllamaChat(model="m")([{"role": "user", "content": "hi"}], tools=None)
        self.assertNotIn("tools", post.call_args[1]["json"])

    def test_errors_become_agent_errors(self):
        import requests

        with mock.patch("requests.post", side_effect=requests.exceptions.ConnectionError("down")):
            with self.assertRaisesRegex(AgentError, "ollama serve"):
                OllamaChat(model="m")([], None)
        missing = mock.Mock(status_code=404, text='{"error":"model \'m\' not found"}')
        with mock.patch("requests.post", return_value=missing):
            with self.assertRaisesRegex(AgentError, "ollama pull m"):
                OllamaChat(model="m")([], None)
        wrong_path = mock.Mock(status_code=404, text="404 page not found")
        with mock.patch("requests.post", return_value=wrong_path):
            with self.assertRaisesRegex(AgentError, "agent.ollama_url"):
                OllamaChat(model="m")([], None)
        garbled = mock.Mock(status_code=200, text="<html>proxy</html>")
        garbled.json.side_effect = ValueError("no json")
        with mock.patch("requests.post", return_value=garbled):
            with self.assertRaisesRegex(AgentError, "non-JSON"):
                OllamaChat(model="m")([], None)
        with mock.patch("requests.post", side_effect=requests.exceptions.ChunkedEncodingError("cut")):
            with self.assertRaisesRegex(AgentError, "request to Ollama failed"):
                OllamaChat(model="m")([], None)


class TestCli(unittest.TestCase):
    def test_cli_prints_answer_sources_and_tools(self):
        from sofc_agent import cli

        scripted = ScriptedChat([
            {"tool_calls": [tool_call("query_dataset", dopant="Al", carbonate="none")]},
            {"content": "30ADC reaches 0.18 S/cm at 500 C [ref04_30ADC]."},
        ])
        with mock.patch.object(cli, "OllamaChat", return_value=scripted), \
                mock.patch("sys.stdout.reconfigure", create=True), \
                mock.patch("builtins.print") as printed:
            code = cli.main(["Al-doped", "ceria?"])
        out = "\n".join(" ".join(str(a) for a in call.args) for call in printed.call_args_list)
        self.assertEqual(code, 0)
        self.assertIn("[ref04_30ADC]", out)
        self.assertIn("Sources:", out)
        self.assertIn("query_dataset(", out)

    def test_cli_reports_backend_errors_with_exit_code_2(self):
        from sofc_agent import cli

        def broken(messages, tools=None):
            raise AgentError("cannot reach Ollama at http://127.0.0.1:11434")

        with mock.patch.object(cli, "OllamaChat", return_value=broken), \
                mock.patch("builtins.print"):
            self.assertEqual(cli.main(["question"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
