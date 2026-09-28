# Evaluation summary

- Model: `qwen2.5:3b-instruct` via Ollama 0.34.4
- Hardware: 11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz, 8 threads, model on CPU only
- Run finished: 2026-09-28 17:48
- Questions: 24 in the set; 24 completed, 0 ended in an error; at most 5 tool calls per question
- Warm-up (model + embedding load, not counted below): 30.9 s

## Headline numbers

| Metric | Result |
| --- | --- |
| Expected tool called, single-tool questions | 20/20 (100%) |
| ...and it was the first tool called | 20/20 (100%) |
| ...and no other tool was called | 20/20 (100%) |
| Both expected tools called, two-tool questions | 3/4 (75%) |
| Answer cites a tool result (answered questions) | 19/19 (100%) |
| ...cited without needing the repair prompt | 19/19 (100%) |
| Answer correct, answerable questions (keyword check) | 13/20 (65%) |
| Correctly declined, unanswerable questions | 4/4 (100%) |
| Wrongly declined, answerable questions | 1/20 (5%) |
| Invented citation tags caught | 6 |
| Answers with numbers not found in any tool output | 1/19 (5%) |
| Time per question: mean / median | 136.3 s / 23.8 s |
| Time per question: min / max | 8.3 s / 2672.8 s |
| Total time | 54.5 min |
| Model calls / tool calls per question (mean) | 2.1 / 1.2 |

## By expected tool (single-tool questions)

| Tool | Questions | Tool matched | Called first | Correct | Cited when answered | Mean time |
| --- | --- | --- | --- | --- | --- | --- |
| `search_literature` | 5 | 5/5 (100%) | 5/5 (100%) | 4/5 (80%) | 4/4 (100%) | 41.4 s |
| `query_dataset` | 5 | 5/5 (100%) | 5/5 (100%) | 3/5 (60%) | 4/4 (100%) | 21.8 s |
| `arrhenius` | 5 | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 4/4 (100%) | 15.2 s |
| `materials_project` | 5 | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 4/4 (100%) | 17.1 s |

## Per question

| ID | Expected | Called | Match | Cites | Correct | Time |
| --- | --- | --- | --- | --- | --- | --- |
| q01 | search_literature | search_literature | yes | yes | yes | 53.2 s |
| q02 | search_literature | search_literature | yes | yes | yes | 30.6 s |
| q03 | search_literature | search_literature | yes | yes | yes | 44.9 s |
| q04 | search_literature | search_literature | yes | yes | no | 70.0 s |
| q05 | search_literature (decline) | search_literature | yes | no | yes | 8.5 s |
| q06 | query_dataset | query_dataset | yes | yes | yes | 26.0 s |
| q07 | query_dataset | query_dataset | yes | yes | no | 27.1 s |
| q08 | query_dataset | query_dataset | yes | yes | yes | 15.4 s |
| q09 | query_dataset | query_dataset | yes | yes | no | 32.0 s |
| q10 | query_dataset (decline) | query_dataset | yes | no | yes | 8.3 s |
| q11 | arrhenius | arrhenius | yes | yes | yes | 18.1 s |
| q12 | arrhenius | arrhenius | yes | yes | yes | 14.7 s |
| q13 | arrhenius | arrhenius | yes | yes | yes | 18.1 s |
| q14 | arrhenius | arrhenius | yes | yes | yes | 15.2 s |
| q15 | arrhenius (decline) | arrhenius | yes | no | yes | 9.7 s |
| q16 | materials_project | materials_project | yes | yes | yes | 21.6 s |
| q17 | materials_project | materials_project | yes | yes | yes | 11.1 s |
| q18 | materials_project | materials_project | yes | yes | yes | 29.7 s |
| q19 | materials_project | materials_project | yes | yes | yes | 11.2 s |
| q20 | materials_project (decline) | materials_project | yes | no | yes | 12.1 s |
| q21 | query_dataset + materials_project | query_dataset + materials_project | yes | yes | no | 35.0 s |
| q22 | query_dataset + search_literature | query_dataset | no | yes | no | 29.2 s |
| q23 | arrhenius + query_dataset | arrhenius + query_dataset | yes | yes | no | 2672.8 s |
| q24 | materials_project + search_literature | materials_project + search_literature | yes | no | no | 55.9 s |

Stop reasons: answered: 19, model_not_found: 1, nothing_to_cite: 4. Nudged (no tool on first try): 2/24 (8%). Repair prompt used: 1/24 (4%).

Correctness is an automatic keyword check against the answer text (see `expected_keywords` in `questions.csv`), not a human grade; it is lenient for the explanatory literature questions. Full answers are in `results.csv`, and every tool call with its arguments is in `traces.jsonl`.
