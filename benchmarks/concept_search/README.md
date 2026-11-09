# Concept search benchmark

Compares ways of choosing the standard concept of a lab test name
([issue #33](https://github.com/precisiondata/aatm/issues/33)):

- `first-result`: the nearest retrieved candidate (`FirstResultSelector`);
- `llm-selector`: an LLM picks among the top 30 candidates (`OpenAILLMSelector`);
- `agent`: the concept search agent (`aatm.concept_search`), given the text with its
  value, unit and sentence;
- `agent-no-context`: the same agent, given the text only.

`lab_results.csv` holds 30 lab results with gold LOINC codes: abbreviations (Hb, K,
PLT, BUN), Portuguese names (creatinina, potássio, ureia, glicemia de jejum), and
urine and 24-hour tests. `also_accept` lists equivalent codes (e.g. platelets with or
without "by Automated count"). Every method searches standard LOINC concepts of the
Measurement domain. Every gold code was checked to exist as a standard Measurement
concept in the vocabulary database.

## Run

From the repository root, after `aatm init` (vocabulary database and embedding index):

```bash
uv run python benchmarks/concept_search/run_benchmark.py --retriever text-embedding-3-small --model openai/gpt-5-mini
```

`--methods` runs a subset; earlier results of the other methods are kept. Results go to
`results/<retriever>__<model>.csv`, one row per text and method, with the agent's
rationale, trace and usage.

## Results (2026-11-09)

`text-embedding-3-small` index, `openai/gpt-5-mini`, `max_tool_calls=8`:

| Method | Correct | LLM calls / text | Input tokens / text | Output tokens / text | Seconds / text |
|---|---|---|---|---|---|
| first-result | 1 / 30 | 0 | – | – | 7.7 |
| llm-selector | 14 / 30 | 1 | not reported | not reported | 13.5 |
| agent-no-context | 27 / 30 | 2.2 | 3,328 | 936 | 66.0 |
| **agent** | **29 / 30** | 2.2 | 3,379 | 903 | 28.0 |

Seconds are per text, with up to 8 agent searches running concurrently. Most of it is
filtered ChromaDB queries (about 5 s each on this 2.5-million-expression index), not
the LLM. The agent used 2.2 tool calls per text on average, at most 5, and never ran
out of budget; 26 answers were *high* confidence and 4 *medium*.

Its one miss: *BUN* → `6299-2` *Urea nitrogen [Mass/volume] in Blood* instead of
`3094-0` *in Serum or Plasma*.
