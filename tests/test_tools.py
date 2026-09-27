"""Test suite for the agent's tools. Standard library only:

    python -m unittest discover -s tests -v

Tests that need the FAISS index are skipped when it is not reachable; the Materials
Project test that needs a key is skipped when .env has none. Everything else always runs.
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
        self.assertTrue(config.rag_root().exists(), f"rag_root missing: {config.rag_root()}")
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
