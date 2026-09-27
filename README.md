# SOFC Research Agent

A small local research assistant for **solid oxide fuel cell (SOFC)** materials. A local
model (`qwen2.5:3b-instruct` served by [Ollama](https://ollama.com)) reads a question and
decides which tool to call: search the paper corpus, query the extracted measurement table,
fit an activation energy, or look up a phase in the Materials Project.

Everything is free. Nothing leaves the machine except the optional Materials Project
lookup, which uses a free API key.

**Status:** the tool layer is built and tested (49 tests, all passing). The model-driven
tool-calling loop is the next step; the tools already expose the JSON schemas it needs.

---

## The four tools

| Tool | Answers | Source |
| --- | --- | --- |
| `search_literature` | "Why do carbonate composites conduct so well below 600 C?" | FAISS index of the paper corpus, reused from the `sofc-rag` project |
| `query_dataset` | "Which Sm-doped carbonate composites were measured at 600 C?" | `dataset.csv`, one row per reported composition, every row carrying its `paper_id` |
| `arrhenius` | "What is Ea for these conductivities at these temperatures?" | closed-form least-squares fit, standard library only |
| `materials_project` | "Is Na2CO3 stable? What is the band gap of CeO2?" | free Materials Project REST API |

Every tool returns a JSON-serialisable dict with:

- `ok` — did it work,
- `summary` — one line the model can read directly,
- the structured detail (passages, rows, fit statistics, material record),
- `error` instead of the detail when something went wrong.

Tools report problems in the return value rather than raising, so a bad call is something
the agent can read and recover from instead of a crash.

---

## Layout

```
config.yaml              paths and knobs (committed)
.env                     API keys (gitignored; copy from .env.example)
sofc_agent/
  config.py              config.yaml + .env loading
  tools/
    __init__.py          TOOLS registry, TOOL_SCHEMAS for tool calling, call_tool()
    literature.py        search_literature
    dataset.py           query_dataset
    arrhenius.py         arrhenius
    materials.py         materials_project
scripts/test_tools.py    runs each tool and prints what it returns
tests/test_tools.py      the test suite (standard library unittest)
```

---

## Setup

### 1. Python environment

`search_literature` needs `faiss-cpu` and `sentence-transformers`; the other three tools
need only the standard library plus `requests` and `PyYAML`. The `sofc-rag` project already
has all of it, so the quickest start is to reuse that interpreter:

```powershell
C:\Users\DELL\Desktop\sofc-rag\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Or create a standalone environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Paths

`config.yaml` points at the two data sources. Edit these if either project moves:

```yaml
paths:
  rag_root: "C:/Users/DELL/Desktop/sofc-rag"            # reads index/chunks.faiss + chunks.jsonl
  dataset_csv: "C:/Users/DELL/Desktop/LLM fc/results/dataset.csv"
```

The index is **read, never rebuilt** here. Rebuild it in `sofc-rag` if the corpus changes;
the embedding model in `config.yaml` must stay the same as the one used to build it.

### 3. Materials Project key

Register free at [materialsproject.org](https://materialsproject.org), copy your key from
[profile.materialsproject.org](https://profile.materialsproject.org/), then:

```powershell
copy .env.example .env
# edit .env:  MP_API_KEY=your_key_here
```

`.env` is gitignored. Without a key the other three tools work normally and
`materials_project` returns a clear `missing_api_key` error instead of failing silently.

### 4. Local model

```powershell
ollama pull qwen2.5:3b-instruct
ollama serve
```

---

## Usage

```python
from sofc_agent.tools import search_literature, query_dataset, arrhenius, materials_project

# 1. literature
hits = search_literature("why carbonate composites conduct below 600 C", k=3)
for p in hits["passages"]:
    print(p["citation"], p["score"], p["snippet"][:80])

# 2. dataset: filter by dopant, carbonate and temperature
rows = query_dataset(dopant="Sm", carbonate="any", temperature_C=600)
print(rows["summary"])

# 3. activation energy from a conductivity series
fit = arrhenius(temperatures=[400, 450, 500, 550, 600],
                conductivities=[0.013, 0.028, 0.055, 0.10, 0.17])
print(fit["activation_energy_eV"], fit["r_squared"])

# 4. stability and band gap
print(materials_project("CeO2")["summary"])
```

Or dispatch by name, the way the agent loop will:

```python
from sofc_agent.tools import TOOL_SCHEMAS, call_tool

call_tool("query_dataset", {"dopant": "Al", "carbonate": "none"})
```

`TOOL_SCHEMAS` is the JSON-schema function list passed straight to Ollama's `tools`
parameter. `call_tool` drops argument names a small model invents rather than raising, and
reports them back in `ignored_arguments`.

---

## Tool notes

**`query_dataset`** — the source columns are free text, so filters are forgiving:
`dopant="Sm"` also matches the co-doped `Ba+Sm` and `Sm+Ca (co-doped)` rows;
`carbonate="any"` / `"none"` splits composites from single-phase oxides;
`carbonate="Na2CO3"` matches `Na2CO3 (amorphous shell)`. Temperatures match within
±25 C by default, so a measurement at 598 C answers a question about 600 C.

**`arrhenius`** — both literature conventions are supported:

- `model="sigma_T"` (default): σ = (A/T)·exp(−Ea/kBT), fits ln(σT) vs 1/T
- `model="sigma"`: σ = A·exp(−Ea/kBT), fits ln(σ) vs 1/T

Either way the slope is −Ea/kB. The two differ by about kB·T̄ (~0.07 eV at 500 C), so the
convention matters when comparing with a published number. The tool flags r² < 0.99,
which in a carbonate composite usually means two conduction regimes rather than bad data.

**`materials_project`** — only ordered DFT structures are indexed, so doped compositions
such as `Ce0.8Sm0.2O2` return no match. Query the parent phases (`CeO2`, `Sm2O3`,
`Na2CO3`) and reason about the composite from those. The most stable polymorph is
returned as `most_stable`, the rest as `polymorphs`.

---

## Testing

```powershell
# assertions
C:\Users\DELL\Desktop\sofc-rag\.venv\Scripts\python.exe -m unittest discover -s tests -v

# see what each tool actually returns
C:\Users\DELL\Desktop\sofc-rag\.venv\Scripts\python.exe scripts\test_tools.py
C:\Users\DELL\Desktop\sofc-rag\.venv\Scripts\python.exe scripts\test_tools.py --tool arrhenius
```

The suite covers filter semantics, numeric coercion, error paths, request construction and
response parsing. The Arrhenius fit is checked by round trip: conductivities are generated
from a known Ea and the fit has to recover it (to 3 decimal places for Ea in 0.20–1.05 eV).
Materials Project parsing is tested against a recorded payload shape, so it runs without a
key; tests that need the FAISS index or a live key skip themselves when unavailable.

---

## Data

The PDFs, the FAISS index and the raw dataset live in the source projects and are not
copied here — only the code that reads them. `paper_id` on every dataset row and
`[Title, p.N]` on every retrieved passage point back to the source, so any answer the
agent gives can be traced to a paper and page.

## License

MIT — see [LICENSE](LICENSE).
