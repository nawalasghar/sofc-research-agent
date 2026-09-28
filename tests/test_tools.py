"""Test suite for the agent's tools. Standard library only:

    python -m unittest discover -s tests -v

The dataset tests here and in test_agent.py need paths.dataset_csv (config.yaml) to point
at dataset.csv from the ceria-carbonate-llm-dataset repository. Tests that need the FAISS
index are skipped when it is not reachable; the Materials Project test that needs a key is
skipped when .env has none. Everything else always runs.
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sofc_agent import config  # noqa: E402
from sofc_agent.tools import TOOLS, TOOL_SCHEMAS, call_tool  # noqa: E402
from sofc_agent.tools.arrhenius import EV_TO_KJ_PER_MOL, KB_EV_PER_K, arrhenius  # noqa: E402
from sofc_agent.tools.dataset import query_dataset  # noqa: E402
from sofc_agent.tools.literature import search_literature  # noqa: E402
from sofc_agent.tools.materials import materials_project, parse_summary_response  # noqa: E402


def synthetic_conductivities(ea_ev: float, prefactor: float, temps_c) -> list[float]:
    """sigma = (A/T) * exp(-Ea / (kB*T)), i.e. exactly the sigma_T model."""
    return [
        prefactor / (t + 273.15) * math.exp(-ea_ev / (KB_EV_PER_K * (t + 273.15)))
        for t in temps_c
    ]


class TestConfig(unittest.TestCase):
    def test_dotted_lookup(self):
        self.assertIsInstance(config.get("literature.default_k"), int)
        self.assertEqual(config.get("nope.not.here", "fallback"), "fallback")

    def test_configured_paths_exist(self):
        # Only the dataset is required; the literature index is optional (its tests skip).
        self.assertTrue(config.dataset_csv().exists(), f"dataset missing: {config.dataset_csv()}")

    def test_dotenv_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "# a comment\n"
                "\n"
                "PLAIN=abc123\n"
                'QUOTED="with spaces"\n'
                "export EXPORTED=xyz\n"
                "NO_EQUALS_SIGN\n"
                "EMPTY=\n",
                encoding="utf-8",
            )
            with mock.patch.object(config, "ENV_FILE", env_file):
                config.load_dotenv.cache_clear()
                values = config.load_dotenv()
        config.load_dotenv.cache_clear()

        self.assertEqual(values["PLAIN"], "abc123")
        self.assertEqual(values["QUOTED"], "with spaces")
        self.assertEqual(values["EXPORTED"], "xyz")
        self.assertEqual(values["EMPTY"], "")
        self.assertNotIn("NO_EQUALS_SIGN", values)

    def test_secrets_are_not_committed(self):
        gitignore = (config.ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env", gitignore)


class TestQueryDataset(unittest.TestCase):
    def test_unfiltered_load(self):
        result = query_dataset()
        self.assertTrue(result["ok"], result)
        self.assertGreater(result["dataset_rows_total"], 0)
        self.assertTrue(all("paper_id" in row for row in result["rows"]))

    def test_dopant_substring_catches_codoped(self):
        result = query_dataset(dopant="Sm")
        dopants = {row["ceria_dopant"] for row in result["rows"]}
        self.assertTrue(any("+" in d for d in dopants), f"no co-doped entry matched: {dopants}")
        self.assertTrue(all("sm" in d.lower() for d in dopants), dopants)

    def test_carbonate_presence_and_absence_partition(self):
        with_carbonate = query_dataset(carbonate="any", limit=100)
        without = query_dataset(carbonate="none", limit=100)
        total = query_dataset(limit=100)["statistics"]["n_matches"]
        self.assertGreater(with_carbonate["statistics"]["n_matches"], 0)
        self.assertGreater(without["statistics"]["n_matches"], 0)
        self.assertEqual(
            with_carbonate["statistics"]["n_matches"] + without["statistics"]["n_matches"],
            total,
            "carbonate=any and carbonate=none must partition the dataset",
        )
        for row in without["rows"]:
            phase = (row["secondary_phase"] or "none").lower()
            self.assertNotIn("co3", phase)
            self.assertNotIn("carbonate", phase)

    def test_carbonate_boolean_matches_string_form(self):
        self.assertEqual(
            query_dataset(carbonate=True, limit=100)["statistics"]["n_matches"],
            query_dataset(carbonate="any", limit=100)["statistics"]["n_matches"],
        )

    def test_specific_carbonate_substring(self):
        result = query_dataset(carbonate="ternary")
        self.assertGreater(result["statistics"]["n_matches"], 0)
        for row in result["rows"]:
            self.assertIn("ternary", row["secondary_phase"].lower())

    def test_temperature_window(self):
        result = query_dataset(temperature_C=600, temperature_tolerance_C=25)
        self.assertGreater(result["statistics"]["n_matches"], 0)
        for row in result["rows"]:
            self.assertIsNotNone(row["conductivity_temp_C"])
            self.assertLessEqual(abs(row["conductivity_temp_C"] - 600), 25)

    def test_temperature_tolerance_is_respected(self):
        tight = query_dataset(temperature_C=575, temperature_tolerance_C=1)["statistics"]["n_matches"]
        loose = query_dataset(temperature_C=575, temperature_tolerance_C=50)["statistics"]["n_matches"]
        self.assertEqual(tight, 0)
        self.assertGreater(loose, 0)

    def test_combined_filters_are_conjunctive(self):
        result = query_dataset(dopant="Sm", carbonate="any", temperature_C=600)
        self.assertTrue(result["ok"], result)
        self.assertGreater(result["statistics"]["n_matches"], 0)
        for row in result["rows"]:
            self.assertIn("sm", row["ceria_dopant"].lower())
            self.assertTrue(
                "co3" in row["secondary_phase"].lower() or "carbonate" in row["secondary_phase"].lower()
            )
            self.assertAlmostEqual(row["conductivity_temp_C"], 600, delta=25)

    def test_numeric_columns_are_numbers(self):
        rows = query_dataset(paper_id="ref11", limit=100)["rows"]
        self.assertTrue(rows)
        for row in rows:
            for field in ("conductivity_S_cm", "conductivity_temp_C", "activation_energy_eV"):
                if row[field] is not None:
                    self.assertIsInstance(row[field], float, f"{field} should be numeric")

    def test_statistics_report_the_best_row(self):
        result = query_dataset(paper_id="ref11", limit=100)
        best = result["statistics"]["best_conductivity"]
        values = [r["conductivity_S_cm"] for r in result["rows"] if r["conductivity_S_cm"] is not None]
        self.assertEqual(best["conductivity_S_cm"], max(values))

    def test_limit_truncates_and_flags(self):
        result = query_dataset(limit=2)
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["returned_rows"], 2)
        self.assertTrue(result["truncated"])

    def test_full_rows_returns_every_column(self):
        curated = query_dataset(limit=1)["rows"][0]
        full = query_dataset(limit=1, full_rows=True)["rows"][0]
        self.assertGreater(len(full), len(curated))
        self.assertIn("notes", full)

    def test_no_match_is_success_with_zero_rows(self):
        result = query_dataset(dopant="Gd")
        self.assertTrue(result["ok"])
        self.assertEqual(result["rows"], [])
        self.assertIn("No rows match", result["summary"])

    def test_bad_arguments_are_reported(self):
        self.assertFalse(query_dataset(temperature_field="colour")["ok"])
        self.assertFalse(query_dataset(temperature_C="hot")["ok"])

    def test_missing_dataset_file_is_reported(self):
        with mock.patch.object(config, "dataset_csv", return_value=Path("Z:/nope/dataset.csv")):
            from sofc_agent.tools import dataset as dataset_mod
            dataset_mod._load_rows.cache_clear()
            result = query_dataset()
        from sofc_agent.tools import dataset as dataset_mod
        dataset_mod._load_rows.cache_clear()
        self.assertFalse(result["ok"])
        self.assertIn("dataset not found", result["error"])


class TestArrhenius(unittest.TestCase):
    def test_recovers_a_known_activation_energy(self):
        temps = [400, 450, 500, 550, 600]
        for ea_true in (0.20, 0.45, 0.61, 1.05):
            sigmas = synthetic_conductivities(ea_true, 1.0e4, temps)
            result = arrhenius(temperatures=temps, conductivities=sigmas)
            self.assertTrue(result["ok"], result)
            self.assertAlmostEqual(result["activation_energy_eV"], ea_true, places=3)
            self.assertAlmostEqual(result["r_squared"], 1.0, places=6)

    def test_recovers_the_prefactor(self):
        temps = [400, 500, 600]
        result = arrhenius(temperatures=temps, conductivities=synthetic_conductivities(0.5, 2500.0, temps))
        self.assertAlmostEqual(result["prefactor_A"], 2500.0, delta=1.0)

    def test_kelvin_and_celsius_agree(self):
        temps_c = [400, 500, 600]
        sigmas = synthetic_conductivities(0.45, 1.0e4, temps_c)
        in_c = arrhenius(temperatures=temps_c, conductivities=sigmas)
        in_k = arrhenius(
            temperatures=[t + 273.15 for t in temps_c], conductivities=sigmas, temperature_unit="K"
        )
        self.assertAlmostEqual(in_c["activation_energy_eV"], in_k["activation_energy_eV"], places=6)

    def test_sigma_model_differs_from_sigma_T_by_the_T_term(self):
        # ln(sigma) = ln(sigma*T) - ln(T), and with x = 1/T the extra term has slope +T,
        # so dropping the 1/T prefactor understates Ea by about kB * T_mean.
        temps = [400, 500, 600]
        sigmas = synthetic_conductivities(0.45, 1.0e4, temps)
        ea_sigma_t = arrhenius(temperatures=temps, conductivities=sigmas)["activation_energy_eV"]
        ea_sigma = arrhenius(temperatures=temps, conductivities=sigmas, model="sigma")["activation_energy_eV"]
        self.assertLess(ea_sigma, ea_sigma_t)
        t_mean = sum(t + 273.15 for t in temps) / len(temps)
        self.assertAlmostEqual(ea_sigma_t - ea_sigma, KB_EV_PER_K * t_mean, delta=0.002)

    def test_unit_conversion_of_the_reported_value(self):
        temps = [400, 500, 600]
        result = arrhenius(temperatures=temps, conductivities=synthetic_conductivities(0.45, 1e4, temps))
        self.assertAlmostEqual(
            result["activation_energy_kJ_per_mol"],
            result["activation_energy_eV"] * EV_TO_KJ_PER_MOL,
            places=1,
        )

    def test_numeric_strings_are_accepted(self):
        result = arrhenius(temperatures=["500", "550", "600"], conductivities=["0.05", "0.1", "0.17"])
        self.assertTrue(result["ok"], result)

    def test_scatter_is_flagged(self):
        result = arrhenius(
            temperatures=[400, 450, 500, 550, 600],
            conductivities=[0.01, 0.09, 0.02, 0.15, 0.03],
        )
        self.assertTrue(result["ok"])
        self.assertLess(result["r_squared"], 0.99)
        self.assertTrue(any("r^2" in note for note in result["notes"]))

    def test_two_points_are_flagged_as_exact(self):
        result = arrhenius(temperatures=[500, 600], conductivities=[0.05, 0.17])
        self.assertTrue(result["ok"])
        self.assertTrue(any("2 points" in note for note in result["notes"]))

    def test_rejects_bad_input(self):
        bad_cases = {
            "length mismatch": arrhenius(temperatures=[500, 550, 600], conductivities=[0.1, 0.2]),
            "too few points": arrhenius(temperatures=[500], conductivities=[0.1]),
            "zero conductivity": arrhenius(temperatures=[500, 600], conductivities=[0.1, 0.0]),
            "negative conductivity": arrhenius(temperatures=[500, 600], conductivities=[0.1, -0.2]),
            "identical temperatures": arrhenius(temperatures=[600, 600], conductivities=[0.1, 0.2]),
            "below absolute zero": arrhenius(temperatures=[-300, 600], conductivities=[0.1, 0.2]),
            "not a number": arrhenius(temperatures=[500, "warm"], conductivities=[0.1, 0.2]),
            "not a list": arrhenius(temperatures=500, conductivities=[0.1]),
            "bad unit": arrhenius(temperatures=[500, 600], conductivities=[0.1, 0.2], temperature_unit="F"),
            "bad model": arrhenius(temperatures=[500, 600], conductivities=[0.1, 0.2], model="linear"),
        }
        for label, result in bad_cases.items():
            with self.subTest(label):
                self.assertFalse(result["ok"], f"{label} should have failed")
                self.assertIn("error", result)


# A hand-built payload in the shape the API returns, so parsing is testable with no key.
FAKE_PAYLOAD = {
    "data": [
        {
            "material_id": "mp-metastable",
            "formula_pretty": "XO2",
            "chemsys": "O-X",
            "is_stable": False,
            "energy_above_hull": 0.035,
            "formation_energy_per_atom": -3.0,
            "band_gap": 0.0,
            "is_gap_direct": False,
            "is_metal": True,
            "symmetry": {"crystal_system": "Tetragonal", "symbol": "P4_2/mnm", "number": 136},
            "density": 6.1,
            "nsites": 6,
            "theoretical": True,
        },
        {
            "material_id": "mp-stable",
            "formula_pretty": "XO2",
            "chemsys": "O-X",
            "is_stable": True,
            "energy_above_hull": 0.0,
            "formation_energy_per_atom": -3.4,
            "band_gap": 2.1234,
            "is_gap_direct": True,
            "is_metal": False,
            "symmetry": {"crystal_system": "Cubic", "symbol": "Fm-3m", "number": 225},
            "density": 7.2,
            "nsites": 3,
            "theoretical": False,
        },
    ],
    "meta": {"total_doc": 2},
}


class TestMaterialsProjectParsing(unittest.TestCase):
    def test_most_stable_entry_wins_regardless_of_payload_order(self):
        result = parse_summary_response(FAKE_PAYLOAD, "XO2")
        self.assertTrue(result["ok"])
        self.assertTrue(result["found"])
        self.assertEqual(result["n_candidates"], 2)
        self.assertEqual(result["most_stable"]["material_id"], "mp-stable")
        self.assertEqual(result["polymorphs"][0]["material_id"], "mp-metastable")

    def test_flattened_fields_and_descriptions(self):
        best = parse_summary_response(FAKE_PAYLOAD, "XO2")["most_stable"]
        self.assertEqual(best["crystal_system"], "Cubic")
        self.assertEqual(best["spacegroup_symbol"], "Fm-3m")
        self.assertEqual(best["band_gap_eV"], 2.1234)
        self.assertEqual(best["mp_url"], "https://materialsproject.org/materials/mp-stable")

    def test_stability_and_gap_wording(self):
        stable = parse_summary_response(FAKE_PAYLOAD, "XO2")
        self.assertIn("convex hull", stable["stability"])
        self.assertIn("direct", stable["band_gap_description"])

        metastable_only = {"data": [FAKE_PAYLOAD["data"][0]]}
        result = parse_summary_response(metastable_only, "XO2")
        self.assertIn("metastable", result["stability"])
        self.assertIn("35 meV/atom", result["stability"])
        self.assertIn("metallic", result["band_gap_description"])

    def test_empty_payload_is_a_clean_no_match(self):
        result = parse_summary_response({"data": []}, "Ce0.8Sm0.2O2")
        self.assertTrue(result["ok"])
        self.assertFalse(result["found"])
        self.assertIsNone(result["most_stable"])
        self.assertIn("parent phase", result["summary"])


class TestMaterialsProjectCall(unittest.TestCase):
    def test_empty_formula_is_rejected_before_any_request(self):
        result = materials_project("")
        self.assertFalse(result["ok"])
        self.assertIn("formula", result["error"])

    def test_missing_key_explains_how_to_fix_it(self):
        with mock.patch.object(config, "secret", return_value=None):
            result = materials_project("CeO2")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "missing_api_key")
        self.assertIn("MP_API_KEY", result["error"])

    def test_unauthorized_response_is_reported(self):
        fake = mock.Mock(status_code=401, text="No API key found in request")
        with mock.patch.object(config, "secret", return_value="wrong-key"), \
                mock.patch("requests.get", return_value=fake):
            result = materials_project("CeO2")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "unauthorized")

    def test_network_failure_is_reported(self):
        with mock.patch.object(config, "secret", return_value="key"), \
                mock.patch("requests.get", side_effect=OSError("dns failure")):
            result = materials_project("CeO2")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "network_error")

    def test_request_is_addressed_and_authenticated_correctly(self):
        fake = mock.Mock(status_code=200)
        fake.json.return_value = FAKE_PAYLOAD
        with mock.patch.object(config, "secret", return_value="test-key"), \
                mock.patch("requests.get", return_value=fake) as get:
            result = materials_project("XO2", limit=5)
        self.assertTrue(result["ok"])
        url, kwargs = get.call_args[0][0], get.call_args[1]
        self.assertTrue(url.endswith("/materials/summary/"), url)
        self.assertEqual(kwargs["headers"]["X-API-KEY"], "test-key")
        self.assertEqual(kwargs["params"]["formula"], "XO2")
        self.assertEqual(kwargs["params"]["_limit"], 5)
        self.assertIn("energy_above_hull", kwargs["params"]["_fields"])

    @unittest.skipUnless(config.secret("MP_API_KEY"), "no MP_API_KEY in .env")
    def test_live_lookup(self):
        result = materials_project("CeO2")
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["found"], result)
        self.assertEqual(result["most_stable"]["formula_pretty"], "CeO2")
        self.assertIsNotNone(result["most_stable"]["band_gap_eV"])


INDEX_READY = all(path.exists() for path in config.index_files())


@unittest.skipUnless(INDEX_READY, "literature index not found; check paths.rag_root")
class TestSearchLiterature(unittest.TestCase):
    def test_returns_cited_passages(self):
        result = search_literature("ceria carbonate composite electrolyte conductivity", k=3)
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(result["passages"]), 3)
        for passage in result["passages"]:
            self.assertTrue(passage["title"])
            self.assertIsNotNone(passage["page"])
            self.assertIn(f"p.{passage['page']}", passage["citation"])
            self.assertTrue(passage["snippet"])

    def test_scores_are_sorted_descending(self):
        scores = [p["score"] for p in search_literature("solid oxide fuel cell anode", k=5)["passages"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_k_is_capped_by_config(self):
        result = search_literature("electrolyte", k=10_000)
        self.assertLessEqual(result["k"], config.get("literature.max_k"))

    def test_empty_query_is_rejected(self):
        self.assertFalse(search_literature("   ")["ok"])

    def test_missing_index_is_reported(self):
        from sofc_agent.tools import literature as lit
        saved = dict(lit._CACHE)
        lit._CACHE.clear()
        try:
            with mock.patch.object(config, "index_files",
                                   return_value=(Path("Z:/no.faiss"), Path("Z:/no.jsonl"))):
                result = search_literature("anything")
        finally:
            lit._CACHE.clear()
            lit._CACHE.update(saved)
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"])


class TestSmallModelInputs(unittest.TestCase):
    """Inputs a 3B model actually produces must still reach the right data."""

    def test_dopant_names_and_acronyms_resolve_to_symbols(self):
        expected = query_dataset(dopant="Sm", limit=100)["statistics"]["n_matches"]
        for spelling in ("sm", "Samarium", "SDC", "Sm-doped ceria", "samarium doped"):
            with self.subTest(spelling):
                self.assertEqual(query_dataset(dopant=spelling, limit=100)["statistics"]["n_matches"], expected)

    def test_dopant_symbols_match_exactly_not_as_substrings(self):
        # 'S' must not match 'Sm'; 'Sm+Ca' needs both; 'co-doped' is not cobalt.
        self.assertEqual(query_dataset(dopant="S")["statistics"]["n_matches"], 0)
        both = query_dataset(dopant="Sm+Ca", limit=100)["rows"]
        self.assertTrue(both)
        self.assertTrue(all("Sm" in r["ceria_dopant"] and "Ca" in r["ceria_dopant"] for r in both))
        co_doped = query_dataset(dopant="co-doped", limit=100)["rows"]
        self.assertGreater(len(co_doped), len(both))  # Ba+Sm and La+Sm are co-doped too
        self.assertTrue(all("+" in r["ceria_dopant"] for r in co_doped))

    def test_no_filter_words_and_real_absences(self):
        total = query_dataset(limit=100)["statistics"]["n_matches"]
        for word in ("none", "any", "all"):
            with self.subTest(word):
                self.assertEqual(query_dataset(dopant=word, limit=100)["statistics"]["n_matches"], total)
        self.assertEqual(query_dataset(dopant="Na")["statistics"]["n_matches"], 0)  # sodium, not "n/a"
        self.assertEqual(query_dataset(dopant="undoped")["statistics"]["n_matches"], 0)
        self.assertEqual(query_dataset(dopant="Gadolinium")["statistics"]["n_matches"], 0)

    def test_temperature_with_units(self):
        plain = query_dataset(temperature_C=600)["statistics"]["n_matches"]
        for spelling in ("600", "600 C", "600C", "about 600 C"):
            with self.subTest(spelling):
                self.assertEqual(query_dataset(temperature_C=spelling)["statistics"]["n_matches"], plain)

    def test_empty_carbonate_string_means_no_filter(self):
        self.assertEqual(
            query_dataset(carbonate="", limit=100)["statistics"]["n_matches"],
            query_dataset(limit=100)["statistics"]["n_matches"],
        )

    def test_arrhenius_accepts_stringified_lists_and_units(self):
        reference = arrhenius([400, 450, 500, 550, 600], [0.013, 0.028, 0.055, 0.10, 0.17])
        for temps, sigmas in [
            ("[400, 450, 500, 550, 600]", "[0.013, 0.028, 0.055, 0.10, 0.17]"),
            ("400, 450, 500, 550, 600", "0.013, 0.028, 0.055, 0.10, 0.17"),
            ("400 450 500 550 600", "0.013 0.028 0.055 0.10 0.17"),
            (["400 C", "450 C", "500 C", "550 C", "600 C"], ["0.013 S/cm", "0.028", "0.055", "0.10", "0.17"]),
        ]:
            with self.subTest(temps=temps):
                result = arrhenius(temps, sigmas)
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["activation_energy_eV"], reference["activation_energy_eV"])

    def test_arrhenius_rejects_booleans(self):
        self.assertFalse(arrhenius([True, 500], [0.1, 0.2])["ok"])

    def test_snippet_window_follows_the_query(self):
        from sofc_agent.tools.literature import _snippet

        text = " ".join(["filler"] * 200 + ["carbonate", "interface", "conduction"] + ["filler"] * 200)
        window = _snippet(text, 20, "interface conduction in carbonate composites")
        self.assertIn("carbonate interface conduction", window)
        self.assertTrue(window.startswith("... ") and window.endswith(" ..."))
        self.assertTrue(_snippet(text, 20, "").startswith("filler"))  # no query -> the start


class TestToolReviewRegressions(unittest.TestCase):
    """Cases found in review (ids in the docstrings)."""

    def labels(self, **filters) -> list[str]:
        return sorted(r["composition_label"].split(" (")[0] for r in query_dataset(limit=100, **filters)["rows"])

    def test_dopant_as_composition_name(self):
        """dopant-1: composition names matched nothing."""
        self.assertEqual(self.labels(dopant="LN-SDC"), ["LN-SDC"])
        self.assertEqual(self.labels(dopant="N-SDC"), ["N-SDC"])  # not LN-SDC or LNK-SDC
        self.assertEqual(self.labels(dopant="NK-CDC"), ["NK-CDC"])
        self.assertEqual(self.labels(dopant="LNK-SDC"), ["LNK-SDC"])
        self.assertEqual(len(self.labels(dopant="CSDC")), 4)

    def test_dopant_as_formula_or_phrase(self):
        """dopant-1: formulas and phrases resolve to the dopant element."""
        sm = query_dataset(dopant="Sm", limit=100)["statistics"]["n_matches"]
        for spelling in ("Ce0.8Sm0.2O1.9", "Sm2O3-doped CeO2", "samaria-doped ceria",
                         "Sm-doped ceria-carbonate", "20 mol% Sm", "Samarium (Sm)"):
            with self.subTest(spelling):
                self.assertEqual(query_dataset(dopant=spelling, limit=100)["statistics"]["n_matches"], sm)
        everything = query_dataset(limit=100)["statistics"]["n_matches"]
        for spelling in ("doped ceria", "all dopants", "any dopant"):
            with self.subTest(spelling):
                self.assertEqual(query_dataset(dopant=spelling, limit=100)["statistics"]["n_matches"], everything)

    def test_several_dopants_mean_either_unless_co_doped(self):
        """dopant-1: 'Ca and Sm' used to return only the co-doped rows."""
        either = query_dataset(dopant="Sm or Ca", limit=100)["statistics"]["n_matches"]
        self.assertGreater(either, query_dataset(dopant="Sm", limit=100)["statistics"]["n_matches"])
        for spelling in ("Sm vs Ca", "Ca and Sm", "Sm, Ca"):
            with self.subTest(spelling):
                self.assertEqual(query_dataset(dopant=spelling, limit=100)["statistics"]["n_matches"], either)
        self.assertEqual(query_dataset(dopant=["Sm", "Ca"], limit=100)["statistics"]["n_matches"], either)
        self.assertEqual(query_dataset(dopant="Sm+Ca", limit=100)["statistics"]["n_matches"], 4)

    def test_unrecognised_dopant_explains_itself(self):
        result = query_dataset(dopant="xyz-123")
        self.assertEqual(result["statistics"]["n_matches"], 0)
        self.assertIn("not an element symbol", result["summary"])

    def test_carbonate_phrases(self):
        """carb-1: common phrasings matched zero rows."""
        self.assertEqual(self.labels(dopant="Al", carbonate="no carbonate"), ["10ADC", "20ADC", "30ADC"])
        self.assertEqual(self.labels(dopant="Al", carbonate="without carbonate"), ["10ADC", "20ADC", "30ADC"])
        with_carbonate = query_dataset(carbonate="any", limit=100)["statistics"]["n_matches"]
        for phrase in ("carbonates", "with carbonate", "carbonate composite", "molten carbonate"):
            with self.subTest(phrase):
                self.assertEqual(query_dataset(carbonate=phrase, limit=100)["statistics"]["n_matches"], with_carbonate)

    def test_carbonate_systems_compare_by_cations(self):
        """carb-1: binary carbonates written with '-', '/', '+' matched nothing."""
        for spelling in ("Na2CO3-K2CO3", "Na2CO3/K2CO3", "Na2CO3 + K2CO3"):
            with self.subTest(spelling):
                self.assertEqual(self.labels(carbonate=spelling), ["NK-CDC", "NK-SDC"])
        for spelling in ("(Li/Na)2CO3", "Li2CO3-Na2CO3", "LiNaCO3"):
            with self.subTest(spelling):
                self.assertEqual(self.labels(carbonate=spelling), ["LN-SDC", "SDC-LiNaCO3"])
        self.assertEqual(self.labels(carbonate="lithium carbonate"), ["L-SDC"])
        self.assertEqual(
            query_dataset(carbonate="sodium carbonate", limit=100)["statistics"]["n_matches"],
            query_dataset(carbonate="Na2CO3", limit=100)["statistics"]["n_matches"],
        )
        # a bare cation means "contains"; it no longer matches words such as "implicit"
        self.assertEqual(self.labels(carbonate="Li"), ["L-SDC", "LK-SDC", "LN-SDC", "LNK-SDC", "SDC-LiNaCO3"])
        self.assertEqual(len(self.labels(carbonate="binary")), 5)

    def test_temperature_in_kelvin_and_ranges(self):
        """temp-1: '873 K' filtered at 873 C; ranges kept only the first number."""
        at_600 = query_dataset(temperature_C=600, limit=100)["statistics"]["n_matches"]
        self.assertEqual(query_dataset(temperature_C="873 K", limit=100)["statistics"]["n_matches"], at_600)
        rows = query_dataset(temperature_C="500-600", limit=100)["rows"]
        temps = {r["conductivity_temp_C"] for r in rows}
        self.assertEqual(temps, {500.0, 550.0, 600.0})
        self.assertEqual(
            query_dataset(temperature_C=[500, 600], limit=100)["statistics"]["n_matches"], len(rows)
        )
        self.assertTrue(all(r["conductivity_temp_C"] >= 550 for r in
                            query_dataset(temperature_C="above 550", limit=100)["rows"]))
        self.assertFalse(query_dataset(temperature_C="400, 500 and 600")["ok"])
        self.assertIn("kelvin", query_dataset(temperature_C=1073)["summary"])

    def test_peak_power_temperature_field(self):
        """temp-2: temperature_field was stripped, so peak-power questions lost rows."""
        from sofc_agent.tools import TOOL_SCHEMAS

        schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "query_dataset")
        self.assertIn("temperature_field", schema["function"]["parameters"]["properties"])
        result = query_dataset(temperature_C=550, temperature_field="pmax_temp_C")
        self.assertEqual(result["statistics"]["best_pmax"]["pmax_mW_cm2"], 1180.0)
        self.assertGreaterEqual(result["statistics"]["n_matches"], 8)

    def test_arrhenius_scientific_notation_is_read_whole(self):
        """arr-1 (tools): '5.2 x 10^-3' was read as 5.2 and gave a wrong, cited Ea."""
        reference = arrhenius([400, 450, 500, 550, 600], [0.0052, 0.011, 0.021, 0.036, 0.058])
        times, minus = "×", "⁻"
        for sigmas in (
            ["5.2 x 10^-3", "1.1 x 10^-2", "2.1 x 10^-2", "3.6 x 10^-2", "5.8 x 10^-2"],
            [f"5.2{times}10{minus}³", f"1.1{times}10{minus}²", f"2.1{times}10{minus}²",
             f"3.6{times}10{minus}²", f"5.8{times}10{minus}²"],
            ["5.2e-3", "1.1e-2", "2.1e-2", "3.6e-2", "5.8e-2"],
            "5.2 x 10^-3, 1.1 x 10^-2, 2.1 x 10^-2, 3.6 x 10^-2, 5.8 x 10^-2",
            ["5.2 mS/cm", "11 mS/cm", "21 mS/cm", "36 mS/cm", "58 mS/cm"],
        ):
            with self.subTest(sigmas=str(sigmas)[:40]):
                result = arrhenius([400, 450, 500, 550, 600], sigmas)
                self.assertTrue(result["ok"], result)
                self.assertAlmostEqual(result["activation_energy_eV"], reference["activation_energy_eV"], places=4)

    def test_arrhenius_unit_on_the_item_wins(self):
        """arr-1 (tools): ['773 K', ...] with the unit argument omitted was read as Celsius."""
        in_kelvin = arrhenius([773, 823, 873], [0.02, 0.05, 0.11], temperature_unit="K")
        self.assertAlmostEqual(
            arrhenius(["773 K", "823 K", "873 K"], [0.02, 0.05, 0.11])["activation_energy_eV"],
            in_kelvin["activation_energy_eV"], places=6,
        )
        self.assertAlmostEqual(
            arrhenius("773 K 823 K 873 K", "0.02 0.05 0.11")["activation_energy_eV"],
            in_kelvin["activation_energy_eV"], places=6,
        )

    def test_arrhenius_rejects_partial_numbers(self):
        for bad in (["5.2 x", "1.1"], ["0.013 S/cmm", "0.02"], [[0.01], [0.02]], [{"a": 1}, 0.02]):
            with self.subTest(bad=bad):
                self.assertFalse(arrhenius([500, 600], bad)["ok"])
        error = arrhenius(["-300", "600"], [0.01, 0.02])["error"]
        self.assertIn("-300", error)  # the parsed value, not one character of the string

    def test_literature_titles_are_unescaped(self):
        from sofc_agent.tools.literature import _short_title

        self.assertEqual(_short_title("SOFCs (300-600&#160;&#176;C)"), "SOFCs (300-600 °C)".replace(" ", " "))

    def test_bug_inside_a_tool_is_not_blamed_on_the_arguments(self):
        from sofc_agent import tools as tools_mod

        def buggy(query: str) -> dict:
            raise TypeError("internal bug")

        with mock.patch.dict(tools_mod.TOOLS, {"search_literature": buggy}):
            result = call_tool("search_literature", {"query": "x"})
            missing = call_tool("search_literature", {})
        self.assertEqual(result["error_code"], "tool_exception")
        self.assertNotIn("bad arguments", result["error"])
        self.assertIn("bad arguments", missing["error"])


class TestRoundTwoToolRegressions(unittest.TestCase):
    def n(self, **filters) -> int:
        return query_dataset(limit=100, **filters)["statistics"]["n_matches"]

    def test_a_list_unit_applies_to_the_whole_list(self):
        """parsing/F2: '773, 823, 873 K' gave a negative Ea (only the last item was K)."""
        reference = arrhenius([773, 823, 873], [0.02, 0.05, 0.11], temperature_unit="K")["activation_energy_eV"]
        for temps in ("773, 823, 873 K", "773 823 873 K", "773 K, 823, 873", ["773", "823", "873 K"]):
            with self.subTest(temps=temps):
                self.assertAlmostEqual(arrhenius(temps, [0.02, 0.05, 0.11])["activation_energy_eV"], reference, places=6)
        q11 = arrhenius([400, 450, 500, 550, 600], [0.013, 0.028, 0.055, 0.10, 0.17])["activation_energy_eV"]
        self.assertAlmostEqual(
            arrhenius([400, 450, 500, 550, 600], "13, 28, 55, 100, 170 mS/cm")["activation_energy_eV"], q11, places=6
        )
        self.assertFalse(arrhenius(["773", "823 K", "873"], [0.02, 0.05, 0.11])["ok"])  # ambiguous

    def test_formulas_and_descriptions_are_not_matched_inside_labels(self):
        """parsing/F3: 'Ce0.8Sm0.2O2' matched one row by substring of its description."""
        sm = self.n(dopant="Sm")
        for spelling in ("Ce0.8Sm0.2O2", "Ce0.8Sm0.2O2-delta", "Ce0.8Sm0.2", "SDC + Na2CO3", "SDC nanocomposite"):
            with self.subTest(spelling):
                self.assertEqual(self.n(dopant=spelling), sm)
        self.assertEqual(self.n(dopant="Ba-SDC"), 1)
        self.assertEqual(self.n(dopant="Sm+Ca in ceria"), 4)

    def test_carbonate_lists_and_phrases(self):
        """parsing/F4."""
        def labels(**filters):
            return sorted(r["composition_label"].split(" (")[0] for r in query_dataset(limit=100, **filters)["rows"])

        either = labels(dopant="Sm", carbonate="Na2CO3 or Li2CO3", temperature_C=600)
        self.assertEqual(either, ["BSDC-carbonate [Sample-3]", "L-SDC", "N-SDC"])
        self.assertEqual(labels(dopant="Sm", carbonate=["Na2CO3", "Li2CO3"], temperature_C=600), either)
        self.assertEqual(labels(carbonate="(Li,Na)2CO3"), ["LN-SDC", "SDC-LiNaCO3"])
        self.assertEqual(labels(carbonate="(Li0.52Na0.48)2CO3"), ["LN-SDC", "SDC-LiNaCO3"])
        self.assertEqual(labels(carbonate="Na2CO3 and K2CO3"), ["NK-CDC", "NK-SDC"])
        self.assertEqual(self.n(dopant="Sm", carbonate="ceria-carbonate composite", temperature_C=600), 8)
        self.assertEqual(self.n(carbonate="no carbonate phase"), self.n(carbonate="none"))
        self.assertEqual(self.n(carbonate="lithium"), self.n(carbonate="Li"))

    def test_temperature_phrasings(self):
        """parsing/F5."""
        at_600 = self.n(dopant="Sm", temperature_C=600)
        for spelling in ("600 +/- 25", "600 ± 25", "600 C plus or minus 25", "600 C (873 K)"):
            with self.subTest(spelling):
                self.assertEqual(self.n(dopant="Sm", temperature_C=spelling), at_600)
        at_least_550 = self.n(dopant="Sm", temperature_C="above 549")
        for spelling in ("550 or higher", "≥ 550", "550+", "at least 550"):
            with self.subTest(spelling):
                self.assertEqual(self.n(dopant="Sm", temperature_C=spelling), at_least_550)

    def test_dopant_lists_co_doping_and_bracketed_ids(self):
        """parsing/F6."""
        self.assertEqual(self.n(dopant="LN-SDC, NK-SDC"), 2)
        self.assertEqual(self.n(dopant="LN-SDC or N-SDC"), 2)
        self.assertEqual(self.n(dopant=["LN-SDC", "N-SDC"]), 2)
        self.assertEqual(self.n(dopant="co-doped ceria"), 8)
        self.assertEqual(self.n(paper_id="[ref11_LN-SDC]"), 7)
        self.assertEqual(self.n(dopant="In-doped ceria"), 0)  # indium: English stopwords are lower case only


class TestRoundThreeToolRegressions(unittest.TestCase):
    def labels(self, **filters):
        return sorted(r["composition_label"].split(" (")[0] for r in query_dataset(limit=100, **filters)["rows"])

    def test_composition_name_inside_a_phrase(self):
        """F2: 'the LNK-SDC composite' matched nothing and the note suggested LN-SDC."""
        self.assertEqual(self.labels(dopant="LNK-SDC composite"), ["LNK-SDC"])
        self.assertEqual(self.labels(dopant="the LNK-SDC composite"), ["LNK-SDC"])
        self.assertEqual(self.labels(dopant="LN-SDC binary-carbonate composite"), ["LN-SDC"])
        self.assertEqual(self.labels(dopant="N-SDC composite"), ["N-SDC"])
        self.assertEqual(len(self.labels(dopant="SDC + Na2CO3")), 21)
        self.assertNotIn("LN-SDC", query_dataset(dopant="xyz-123")["summary"])

    def test_paper_id_none_means_no_filter(self):
        everything = query_dataset(limit=100)["statistics"]["n_matches"]
        for value in ("none", "null", "any"):
            with self.subTest(value):
                self.assertEqual(query_dataset(paper_id=value, limit=100)["statistics"]["n_matches"], everything)

    def test_s_cm_minus_one_is_read_as_a_unit(self):
        from sofc_agent.numbers import quantities

        for unit in ("S cm-1", "S cm^-1", "S cm⁻¹", "S·cm⁻¹", "Scm-1", "mS cm-1"):
            with self.subTest(unit):
                self.assertEqual([k for _, k, _ in quantities(f"0.013 {unit} at 400 C")], ["conductivity", "temperature"])
        self.assertTrue(arrhenius([400, 600], ["0.013 S cm-1", "0.17 S cm-1"])["ok"])


class TestRegistry(unittest.TestCase):
    def test_every_tool_has_a_matching_schema(self):
        schema_names = {entry["function"]["name"] for entry in TOOL_SCHEMAS}
        self.assertEqual(schema_names, set(TOOLS))

    def test_schema_arguments_exist_on_the_functions(self):
        import inspect
        for entry in TOOL_SCHEMAS:
            name = entry["function"]["name"]
            params = set(entry["function"]["parameters"]["properties"])
            signature = set(inspect.signature(TOOLS[name]).parameters)
            with self.subTest(name):
                self.assertTrue(params <= signature, f"{params - signature} not accepted by {name}")
                required = set(entry["function"]["parameters"].get("required", []))
                self.assertTrue(required <= params)

    def test_dispatch(self):
        result = call_tool("arrhenius", {"temperatures": [500, 600], "conductivities": [0.05, 0.17]})
        self.assertTrue(result["ok"], result)

    def test_unknown_tool(self):
        result = call_tool("predict_the_future", {})
        self.assertFalse(result["ok"])
        self.assertIn("unknown tool", result["error"])

    def test_hallucinated_arguments_are_dropped_not_fatal(self):
        result = call_tool("query_dataset", {"dopant": "Sm", "colour": "blue", "pressure": 1})
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["ignored_arguments"], ["colour", "pressure"])

    def test_missing_required_argument(self):
        result = call_tool("arrhenius", {"temperatures": [500, 600]})
        self.assertFalse(result["ok"])
        self.assertIn("bad arguments", result["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
