"""Tests for evaluation scoring, the question set, and crash-safe incremental results.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sofc_agent.agent import NOT_FOUND, AgentResult, Source, ToolCallRecord  # noqa: E402
from sofc_agent.evaluation import (  # noqa: E402
    RESULT_FIELDS,
    keyword_match,
    load_questions,
    render_summary,
    score,
    summarize,
)
from sofc_agent.tools import TOOLS  # noqa: E402


def fake_result(tools=("query_dataset",), body="LN-SDC reaches 0.31 S/cm [ref11_LN-SDC].",
                cited=True, not_found=False, repaired=False, sources_note="") -> AgentResult:
    calls = [ToolCallRecord(t, {}, True, "ok", 0.01) for t in tools]
    cited_sources = [Source("ref11_LN-SDC", "query_dataset", "ref11_LN-SDC", "ref11")] if cited else []
    return AgentResult(
        question="q", answer=body + sources_note, answer_body=body, model_answer=body,
        tool_calls=calls, cited_sources=cited_sources, invalid_citations=[],
        ungrounded_numbers=[], cited=cited, not_found=not_found, nudged=False,
        repaired=repaired, stop_reason="answered", llm_calls=2, prompt_tokens=10,
        completion_tokens=5, elapsed_s=12.34, model="m",
    )


def question(**overrides) -> dict:
    row = {"id": "q", "category": "single", "expected_tools": "query_dataset",
           "expected_behavior": "answer", "question": "?", "expected_keywords": "0.31;LN-SDC"}
    row.update(overrides)
    return row


class TestKeywordMatch(unittest.TestCase):
    def test_groups_and_alternatives(self):
        self.assertTrue(keyword_match("Ea is 0.717 eV", "0.717"))
        self.assertTrue(keyword_match("Ea is 0.72 eV", "0.717"))
        self.assertFalse(keyword_match("Ea is 0.66 eV", "0.717"))
        self.assertTrue(keyword_match("0.13, 0.15 and 0.18 S/cm", "0.13;0.15;0.18"))
        self.assertFalse(keyword_match("0.13 and 0.18 S/cm", "0.13;0.15;0.18"))
        self.assertIsNone(keyword_match("anything", ""))

    def test_case_and_unicode_dashes(self):
        self.assertTrue(keyword_match("space group IA-3", "ia-3|ia3"))
        self.assertTrue(keyword_match("space group Ia−3", "ia-3"))
        self.assertFalse(keyword_match("space group Ia-3d", "Ia-3"))  # garnet, a different group


class TestScore(unittest.TestCase):
    def test_correct_cited_single_tool_answer(self):
        row = score(question(), fake_result())
        self.assertEqual(row["tool_match"], "yes")
        self.assertEqual(row["first_tool_match"], "yes")
        self.assertEqual(row["extra_tools"], "")
        self.assertEqual(row["cites_source"], "yes")
        self.assertEqual(row["answer_correct"], "yes")
        self.assertEqual(row["time_s"], "12.3")
        self.assertEqual(set(row), set(RESULT_FIELDS))

    def test_wrong_tool_and_extra_tool(self):
        row = score(question(), fake_result(tools=("search_literature", "query_dataset")))
        self.assertEqual(row["tool_match"], "yes")
        self.assertEqual(row["first_tool_match"], "no")
        self.assertEqual(row["extra_tools"], "search_literature")
        row = score(question(), fake_result(tools=("search_literature",)))
        self.assertEqual(row["tool_match"], "no")

    def test_two_tool_question_needs_both(self):
        q = question(category="two_tool", expected_tools="query_dataset;materials_project")
        self.assertEqual(score(q, fake_result(tools=("query_dataset",)))["tool_match"], "no")
        self.assertEqual(score(q, fake_result(tools=("materials_project", "query_dataset")))["tool_match"], "yes")

    def test_keywords_ignore_the_sources_list(self):
        """A wrong number in the answer must not pass because a source line has the right one."""
        result = fake_result(body="Ea is 0.66 eV [Arrhenius fit].",
                             sources_note="\n\nSources:\n- Arrhenius fit: Ea = 0.7170 eV")
        self.assertEqual(score(question(expected_keywords="0.717|0.72"), result)["answer_correct"], "no")

    def test_a_decline_on_an_answerable_question_is_wrong(self):
        result = fake_result(body=NOT_FOUND, cited=False, not_found=True)
        self.assertEqual(score(question(expected_keywords="0.31"), result)["answer_correct"], "no")

    def test_decline_questions(self):
        q = question(expected_behavior="decline", expected_keywords="")
        declined = fake_result(body=NOT_FOUND, cited=False, not_found=True)
        self.assertEqual(score(q, declined)["answer_correct"], "yes")
        in_words = fake_result(body="There are no Gd-doped rows in the dataset.", cited=False)
        self.assertEqual(score(q, in_words)["answer_correct"], "yes")
        guessed = fake_result(body="GDC reaches 0.1 S/cm [ref11_LN-SDC].")
        self.assertEqual(score(q, guessed)["answer_correct"], "no")

    def test_decline_that_still_states_a_value_is_wrong(self):
        """decline-1: 'there is no X, but X reaches 0.31 S/cm' passed as a decline."""
        q = question(expected_behavior="decline", expected_keywords="")
        mixed = fake_result(body="There is no separate GDC entry, but gadolinium-doped ceria reaches "
                                 "0.31 S/cm at 600 C [ref11_LN-SDC].")
        self.assertEqual(score(q, mixed)["answer_correct"], "no")
        honest = fake_result(body="The papers do not report a melting point for gold nanoparticles; "
                                  "they only discuss carbonate melting near 450 C [L1].")
        self.assertEqual(score(q, honest)["answer_correct"], "yes")
        counted = fake_result(body="None of the 25 rows is Gd-doped.", cited=False)
        self.assertEqual(score(q, counted)["answer_correct"], "yes")

    def test_round_two_scoring(self):
        """eval/F1-F4."""
        questions = {q["id"]: q for q in load_questions(ROOT / "eval" / "questions.csv")}
        # F1: the tool prints f"{2.0035:.3f}" = '2.003'
        self.assertTrue(keyword_match("2.003 eV indirect band gap", questions["q16"]["expected_keywords"].split(";")[0]))
        # F2: inflections of word alternatives
        self.assertTrue(keyword_match("The shell coats the SDC core and does not permit electrons to move [L3].",
                                      questions["q04"]["expected_keywords"]))
        self.assertTrue(keyword_match("fast transport along the ceria-carbonate interfaces [L1]",
                                      questions["q01"]["expected_keywords"]))
        # F3: stability, scoped to the clause naming the formula
        q16 = questions["q16"]["expected_keywords"]
        self.assertTrue(keyword_match("CeO2 (Fm-3m) is on the convex hull with a 2.0 eV gap; "
                                      "the other polymorphs are not on the convex hull.", q16))
        self.assertFalse(keyword_match("CeO2 is metastable, 119 meV/atom above the convex hull, with a 2.00 eV band gap.", q16))
        self.assertFalse(keyword_match("No, Li2CO3 is metastable; band gap 5.057 eV [MP1]",
                                       questions["q19"]["expected_keywords"]))
        self.assertFalse(keyword_match("Li2CO3 does not lie on the convex hull; its band gap is 5.06 eV",
                                       questions["q19"]["expected_keywords"]))

    def test_round_two_declines(self):
        """eval/F4: honest declines that restate the question pass; guesses do not."""
        questions = {q["id"]: q for q in load_questions(ROOT / "eval" / "questions.csv")}

        def decline(qid, body, ungrounded=()):
            result = fake_result(body=body)
            result.ungrounded_numbers = list(ungrounded)
            return score(questions[qid], result)["answer_correct"]

        self.assertEqual(decline("q15", "The activation energy cannot be determined from a single conductivity value "
                                        "of 0.05 S/cm at 600 C; at least two temperatures are needed [L1]."), "yes")
        self.assertEqual(decline("q20", "The Materials Project does not index the doped composition Ce0.8Sm0.2O2; its "
                                        "parent phase CeO2 has a 2.003 eV band gap [MP1]."), "yes")
        self.assertEqual(decline("q05", "The papers do not report it; gold nanoparticles melt at about 1064 C [L1]."), "no")
        self.assertEqual(decline("q10", "The dataset has no GDC rows; GDC is typically about 0.1 at 600 C.",
                                 ungrounded=["0.1"]), "no")

    def test_round_three_scoring(self):
        """eval-R3-1/2/3 and F4."""
        questions = {q["id"]: q for q in load_questions(ROOT / "eval" / "questions.csv")}

        def decline(qid, body):
            return score(questions[qid], fake_result(body=body))["answer_correct"]

        for body in (
            "No Materials Project entry matches the formula Ce0.8Sm0.2O2. The parent phase CeO2 has a 2.003 eV band gap [MP1].",
            "Ce0.8Sm0.2O2 is not listed in the Materials Project; CeO2 has a 2.003 eV gap [MP1].",
            "The Materials Project doesn’t have an entry for Ce0.8Sm0.2O2; CeO2 has 2.003 eV [MP1].",
            "The Materials Project does not include the doped composition; CeO2 has a 2.003 eV gap [MP1].",
        ):
            with self.subTest(body[:50]):
                self.assertEqual(decline("q20", body), "yes")
        self.assertEqual(decline("q20", "Ce0.8Sm0.2O2 has a band gap of 2.003 eV [MP1]."), "no")
        self.assertEqual(decline("q15", "At least 2 temperatures are needed to calculate the activation energy; "
                                        "the question gives only one conductivity value [L1]."), "yes")

        literature = Source("L1", "search_literature", "Title, p.1", "Title, p.1")
        dataset = Source("ref11_LN-SDC", "query_dataset", "ref11_LN-SDC", "ref11")
        half = fake_result(tools=("query_dataset", "search_literature"),
                           body="The dataset reports 0.45 eV for LN-SDC [ref11_LN-SDC].")
        self.assertEqual(score(questions["q22"], half)["answer_correct"], "no")
        full = fake_result(tools=("query_dataset", "search_literature"),
                           body="LN-SDC has 0.45 eV [ref11_LN-SDC]; the binary carbonate melts earlier [Title, p.1].")
        full.cited_sources = [dataset, literature]
        self.assertEqual(score(questions["q22"], full)["answer_correct"], "yes")

    def test_numeric_keywords_follow_rounding_not_substrings(self):
        """kw-1 and kw-2."""
        self.assertTrue(keyword_match("5.057 eV [MP1]", "5.0574"))
        self.assertTrue(keyword_match("about 5.1 eV", "5.0574"))
        self.assertFalse(keyword_match("5 eV", "5.0574"))
        self.assertTrue(keyword_match("0.807 eV", "0.8065"))
        self.assertFalse(keyword_match("Ea = 0.453 eV", "0.45"))
        self.assertFalse(keyword_match("NK-SDC has 0.31 S/cm [ref11_LN-SDC]", "0.31;LN-SDC"))
        self.assertFalse(keyword_match("Na2CO3 is unstable", "stable|hull;!unstable|not stable"))
        self.assertTrue(keyword_match("Na2CO3 is on the convex hull", "stable|hull;!unstable|not stable"))


class TestSummary(unittest.TestCase):
    def test_summary_numbers_and_markdown(self):
        rows = [
            score(question(id="a"), fake_result()),
            score(question(id="b"), fake_result(tools=("search_literature",), cited=False, body="x")),
            score(question(id="c", expected_behavior="decline", expected_keywords=""),
                  fake_result(body=NOT_FOUND, cited=False, not_found=True)),
            {**{f: "" for f in RESULT_FIELDS}, "id": "d", "category": "single",
             "expected_tools": "arrhenius", "expected_behavior": "answer", "time_s": "600.0",
             "question": "?", "error": "Ollama did not answer within 600 s"},
        ]
        stats = summarize(rows)
        self.assertEqual(stats["n_completed"], 3)
        self.assertEqual(stats["n_errors"], 1)
        self.assertEqual(stats["single_tool_match"], (2, 3))
        self.assertEqual(stats["cited_when_answered"], (1, 2))
        self.assertEqual(stats["answerable_correct"], (1, 2))
        self.assertEqual(stats["declines_correct"], (1, 1))
        self.assertAlmostEqual(stats["time_total_s"], 36.9)

        text = render_summary(stats, rows, {"model": "m", "max_tool_calls": 5, "warmup_s": 3.0})
        self.assertIn("| Expected tool called, single-tool questions | 2/3 (67%) |", text)
        self.assertIn("error: Ollama did not answer", text)
        table = text.split("## Headline numbers")[1].split("\n\n")[1]
        self.assertTrue(all(line.startswith("|") for line in table.splitlines()),
                        "a blank or stray line would break the Markdown table")


class TestQuestionSet(unittest.TestCase):
    def setUp(self):
        self.questions = load_questions(ROOT / "eval" / "questions.csv")

    def test_design_24_questions_5_per_tool_plus_4_two_tool(self):
        self.assertEqual(len(self.questions), 24)
        single = [q for q in self.questions if q["category"] == "single"]
        two = [q for q in self.questions if q["category"] == "two_tool"]
        self.assertEqual(len(single), 20)
        self.assertEqual(len(two), 4)
        for tool in TOOLS:
            with self.subTest(tool):
                self.assertEqual(sum(1 for q in single if q["expected_tools"] == tool), 5)
        for q in two:
            tools = q["expected_tools"].split(";")
            self.assertEqual(len(tools), 2, q["id"])
            self.assertEqual(len(set(tools)), 2, q["id"])

    def test_every_expected_tool_exists(self):
        for q in self.questions:
            for tool in q["expected_tools"].split(";"):
                self.assertIn(tool, TOOLS, q["id"])

    def test_answer_questions_have_keywords(self):
        for q in self.questions:
            with self.subTest(q["id"]):
                if q["expected_behavior"] == "answer":
                    positives = [g for g in q["expected_keywords"].split(";") if g.strip() and not g.strip().startswith("!")]
                    self.assertTrue(positives, "an answer question needs something to check")

    def test_keywords_are_not_given_away_by_the_question(self):
        """A keyword the question already contains would pass for any answer that repeats it."""
        from sofc_agent.evaluation import _groups, _hit

        for q in self.questions:
            for forbidden, alternatives, _ in _groups(q["expected_keywords"]):
                if forbidden or any(a.replace(".", "").isdigit() for a in alternatives):
                    continue
                with self.subTest(q["id"], alternatives=alternatives):
                    self.assertFalse(_hit(alternatives, q["question"]), "keyword appears in the question")

    def test_arrhenius_questions_carry_their_own_data(self):
        """The agent refuses fits on numbers absent from the question, so each Arrhenius
        question must state its data or be a deliberate decline."""
        from sofc_agent.numbers import numbers_in

        for q in self.questions:
            if "arrhenius" in q["expected_tools"] and q["expected_behavior"] == "answer":
                self.assertGreaterEqual(len(numbers_in(q["question"])), 4, q["id"])


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestIncrementalResults(unittest.TestCase):
    """A crash mid-run must keep every finished question, and a rerun must not redo them."""

    def setUp(self):
        self.runner = _load_runner()
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.questions_path = tmp / "questions.csv"
        with self.questions_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "category", "expected_tools", "expected_behavior",
                                                   "question", "expected_keywords"])
            writer.writeheader()
            for i in range(1, 5):
                writer.writerow({"id": f"q{i}", "category": "single", "expected_tools": "query_dataset",
                                 "expected_behavior": "answer", "question": f"question {i}",
                                 "expected_keywords": "0.31"})
        self.patches = [
            mock.patch.object(self.runner, "QUESTIONS", self.questions_path),
            mock.patch.object(self.runner, "RESULTS", tmp / "results.csv"),
            mock.patch.object(self.runner, "TRACES", tmp / "traces.jsonl"),
            mock.patch.object(self.runner, "SUMMARY", tmp / "summary.md"),
            mock.patch.object(self.runner, "META", tmp / "run_meta.json"),
            mock.patch.object(self.runner, "warm_up", return_value=0.1),
            mock.patch.object(self.runner, "_ollama_info", return_value=("0.0", "CPU only")),
            mock.patch.object(self.runner, "OllamaChat", return_value=mock.Mock(model="m", url="u")),
            mock.patch.object(sys, "argv", ["run_eval.py"]),
            mock.patch("builtins.print"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def _agent(self, crash_on: str | None = None, error_on: str | None = None):
        asked: list[str] = []

        class FakeAgent:
            max_tool_calls = 5

            def __init__(self, chat=None):
                pass

            def ask(self, text):
                asked.append(text)
                if crash_on and text == crash_on:
                    raise KeyboardInterrupt  # the process dies mid-question
                if error_on and text == error_on:
                    raise self_module.AgentError("Ollama did not answer within 600 s")
                return fake_result()

        self_module = self.runner
        return FakeAgent, asked

    def test_crash_keeps_finished_rows_and_resume_skips_them(self):
        crashing, asked = self._agent(crash_on="question 3")
        with mock.patch.object(self.runner, "Agent", crashing):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.main()
        self.assertEqual(asked, ["question 1", "question 2", "question 3"])
        self.assertEqual([r["id"] for r in self.runner.read_results()], ["q1", "q2"])
        traces = (Path(self.tmp.name) / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(t)["id"] for t in traces], ["q1", "q2"])

        healthy, asked_again = self._agent()
        with mock.patch.object(self.runner, "Agent", healthy):
            self.assertEqual(self.runner.main(), 0)
        self.assertEqual(asked_again, ["question 3", "question 4"], "finished questions must not rerun")
        self.assertEqual([r["id"] for r in self.runner.read_results()], ["q1", "q2", "q3", "q4"])
        self.assertTrue((Path(self.tmp.name) / "summary.md").exists())

    def test_backend_errors_are_recorded_then_retried(self):
        failing, _ = self._agent(error_on="question 2")
        with mock.patch.object(self.runner, "Agent", failing):
            self.runner.main()
        rows = {r["id"]: r for r in self.runner.read_results()}
        self.assertIn("did not answer", rows["q2"]["error"])

        healthy, asked = self._agent()
        with mock.patch.object(self.runner, "Agent", healthy):
            self.runner.main()
        self.assertEqual(asked, ["question 2"])
        rows = self.runner.read_results()
        self.assertEqual(sorted(r["id"] for r in rows), ["q1", "q2", "q3", "q4"])
        self.assertTrue(all(r["error"] == "" for r in rows))

    def _agent_with_tool_error(self, code: str, on: str):
        asked: list[str] = []

        class FlakyAgent:
            max_tool_calls = 5

            def __init__(self, chat=None):
                pass

            def ask(self, text):
                asked.append(text)
                result = fake_result()
                if text == on:
                    result.tool_calls[0].ok = False
                    result.tool_calls[0].error_code = code
                return result

        return FlakyAgent, asked

    def test_tool_infrastructure_failures_are_retried_not_scored(self):
        """infra-1: an HTTP 429 from Materials Project was scored as a wrong answer."""
        flaky, _ = self._agent_with_tool_error("rate_limited", on="question 2")
        with mock.patch.object(self.runner, "Agent", flaky):
            self.runner.main()
        rows = {r["id"]: r for r in self.runner.read_results()}
        self.assertIn("rate_limited", rows["q2"]["error"])

        healthy, asked = self._agent()
        with mock.patch.object(self.runner, "Agent", healthy):
            self.runner.main()
        self.assertEqual(asked, ["question 2"])

    def test_model_argument_errors_are_scored(self):
        """A bad formula (HTTP 400) is the model's doing and stays a scored row."""
        flaky, _ = self._agent_with_tool_error("http_400", on="question 2")
        with mock.patch.object(self.runner, "Agent", flaky):
            self.runner.main()
        rows = {r["id"]: r for r in self.runner.read_results()}
        self.assertEqual(rows["q2"]["error"], "")

    def test_missing_final_newline_does_not_merge_rows(self):
        """resume-1 (a): a crash that cut the final newline merged the next row into it."""
        crashing, _ = self._agent(crash_on="question 3")
        with mock.patch.object(self.runner, "Agent", crashing):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.main()
        results = Path(self.tmp.name) / "results.csv"
        results.write_text(results.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")

        healthy, asked = self._agent()
        with mock.patch.object(self.runner, "Agent", healthy):
            self.runner.main()
        self.assertEqual(asked, ["question 3", "question 4"])
        rows = self.runner.read_results()
        self.assertEqual([r["id"] for r in rows], ["q1", "q2", "q3", "q4"])
        self.assertTrue(all(self.runner.finished(r) for r in rows))

    def test_one_trace_per_question_after_retries(self):
        """resume-1 (b): retried questions left duplicate traces."""
        crashing, _ = self._agent(crash_on="question 3")
        with mock.patch.object(self.runner, "Agent", crashing):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.main()
        traces = Path(self.tmp.name) / "traces.jsonl"
        traces.write_text(traces.read_text(encoding="utf-8") + '{"id": "q3", "partial": true}\n{"id": "q',
                          encoding="utf-8")
        healthy, _ = self._agent()
        with mock.patch.object(self.runner, "Agent", healthy):
            self.runner.main()
        ids = [json.loads(line)["id"] for line in traces.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sorted(ids), ["q1", "q2", "q3", "q4"])

    def test_summary_survives_torn_rows_and_removed_questions(self):
        """resume-1 (c, d)."""
        healthy, _ = self._agent()
        with mock.patch.object(self.runner, "Agent", healthy):
            self.runner.main()
        results = Path(self.tmp.name) / "results.csv"
        results.write_text(results.read_text(encoding="utf-8") + "q9,single,query_dat", encoding="utf-8")
        with mock.patch.object(sys, "argv", ["run_eval.py", "--summary-only"]):
            self.assertEqual(self.runner.main(), 0)
        summary = (Path(self.tmp.name) / "summary.md").read_text(encoding="utf-8")
        self.assertIn("Questions: 4 in the set; 4 completed, 0 ended in an error;", summary)

    def test_rerun_and_unknown_ids(self):
        healthy, _ = self._agent()
        with mock.patch.object(self.runner, "Agent", healthy):
            self.runner.main()
        again, asked = self._agent()
        with mock.patch.object(self.runner, "Agent", again), \
                mock.patch.object(sys, "argv", ["run_eval.py", "--only", "q2", "--rerun"]):
            self.runner.main()
        self.assertEqual(asked, ["question 2"])
        self.assertEqual(sorted(self.runner.current_results()), ["q1", "q2", "q3", "q4"])
        with mock.patch.object(sys, "argv", ["run_eval.py", "--only", "q99"]):
            self.assertEqual(self.runner.main(), 2)

    def test_failed_rerun_keeps_the_old_answer(self):
        """eval/F5 (a): --rerun used to delete the finished row before the new one existed."""
        healthy, _ = self._agent()
        with mock.patch.object(self.runner, "Agent", healthy):
            self.runner.main()
        with mock.patch.object(self.runner, "warm_up", side_effect=self.runner.AgentError("cannot reach Ollama")), \
                mock.patch.object(sys, "argv", ["run_eval.py", "--only", "q2", "--rerun"]):
            self.assertEqual(self.runner.main(), 2)
        current = self.runner.current_results()
        self.assertEqual(sorted(current), ["q1", "q2", "q3", "q4"])
        self.assertEqual(current["q2"]["error"], "")
        failing, _ = self._agent(error_on="question 2")
        with mock.patch.object(self.runner, "Agent", failing), \
                mock.patch.object(sys, "argv", ["run_eval.py", "--only", "q2", "--rerun"]):
            self.runner.main()
        self.assertEqual(self.runner.current_results()["q2"]["error"], "", "an errored rerun keeps the answer")

    def test_only_keeps_other_questions_errors_and_reports_missing(self):
        """eval/F5 (b): --only dropped other questions' error rows; the summary hid them."""
        flaky, _ = self._agent_with_tool_error("rate_limited", on="question 4")
        with mock.patch.object(self.runner, "Agent", flaky):
            self.runner.main()
        healthy, _ = self._agent()
        with mock.patch.object(self.runner, "Agent", healthy), \
                mock.patch.object(sys, "argv", ["run_eval.py", "--only", "q1", "--rerun"]):
            self.runner.main()
        self.assertIn("rate_limited", self.runner.current_results()["q4"]["error"])
        summary = (Path(self.tmp.name) / "summary.md").read_text(encoding="utf-8")
        self.assertIn("3 completed, 1 ended in an error", summary)

    def test_torn_last_line_is_rerun(self):
        healthy, _ = self._agent()
        with mock.patch.object(self.runner, "Agent", healthy):
            self.runner.main()
        results = Path(self.tmp.name) / "results.csv"
        lines = results.read_text(encoding="utf-8").splitlines()
        results.write_text("\n".join(lines[:-1]) + "\nq4,single,query_dat", encoding="utf-8")

        again, asked = self._agent()
        with mock.patch.object(self.runner, "Agent", again):
            self.runner.main()
        self.assertEqual(asked, ["question 4"])
        self.assertEqual([r["id"] for r in self.runner.read_results()], ["q1", "q2", "q3", "q4"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
