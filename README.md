# rerank-eval

A small repository for determining, under the same conditions every time with frozen data, whether **reranking**—using a judgment model to reorder the top candidates from vector search—actually improves search quality.

It accompanies the blog post, “Does reranking vector-search results with Jev really make them better?” The repository includes results from three approaches: the Jev 1.13.0 judgment model, Cloudflare Clef, and an LLM (`gpt-6-luna`) making Yes/No judgments.

## Contents

| Path | Description |
|---|---|
| `data/corpus.json` | 41 English documents for the fictional company “Larkspur Analytics.” They were written specifically for this repository and contain no third-party copyrighted material. |
| `data/queries.json` | 60 tuning questions across five categories. Answers are stored as **quoted passages** from the documents and relevance grades (3/2/1). |
| `data/pools.json` | The 150 candidates for each question, frozen in their original vector-search order (221 chunks). |
| `results/jev-1.13.0.json` | The probability Jev 1.13.0 assigned to each candidate being useful (150 candidates × 60 questions = 9,000). |
| `results/clef.json` | The probability returned by Clef (60 candidates × 60 questions = 3,600; 12 timed out after five seconds). |
| `results/gpt-6-luna-medium.json` | Yes/No results from gpt-6-luna at medium reasoning effort (60 candidates × 60 questions = 3,600; four received no response). |
| `src/rerank_eval/metrics.py` | Metrics: coverage matching, nDCG, Recall, and paired bootstrap. Standard library only. |
| `src/rerank_eval/evaluate.py` | Scores the original and reranked orders using the same rules, then reports the adoption gates. |
| `src/rerank_eval/run.py` | Queries Jev or an LLM one candidate at a time, with a spending cap and resumability. |

> [!NOTE]
> The 60 evaluation questions and their answers are not included. They are held back for adoption decisions.

## Try it now (no API key required)

```sh
uv run --extra dev pytest -q                      # Hand-worked metric examples and reproduction of included Jev/LLM results
PYTHONPATH=src python3 -m rerank_eval.evaluate results/jev-1.13.0.json
PYTHONPATH=src python3 -m rerank_eval.evaluate results/jev-1.13.0.json --top-k 60   # Compare with the same candidate set as Clef and the LLM
PYTHONPATH=src python3 -m rerank_eval.evaluate results/clef.json
PYTHONPATH=src python3 -m rerank_eval.evaluate results/gpt-6-luna-medium.json
```

`evaluate` reports:

- `overall` / `categories`: nDCG@10, Recall@12, and Recall for selected context candidates, before and after reranking.
- `bootstrap_95`: a 95% interval for the nDCG@10 difference (10,000 iterations; seed `20261006`).
- `gates`: preregistered adoption gates: mean improvement ≥ 0.03, interval lower bound > 0, category degradation ≤ 0.02, and no overall or per-category Recall decline.
- `top_k_prefixes`: results when reranking only the top 20/40/60/100/150 candidates.
- `relevance_floor`: context Recall and precision as the probability threshold changes from 0 to 1.

Results on the tuning data (all three compared on the same top 60 candidates):

| | Original order | Jev 1.13.0 | Clef | gpt-6-luna medium |
|---|---:|---:|---:|---:|
| Output | | Probability | Probability | Yes/No |
| nDCG@10 | 0.753 | **0.919** | 0.851 | 0.870 |
| nDCG@10 difference (95% interval) | | **+0.165** (0.098–0.239) | +0.098 (0.022–0.172) | +0.116 (0.065–0.172) |
| Recall@12 | 0.935 | 0.977 | 0.944 | 0.960 |
| Gate | | All passed | Failed Recall (follow-up 1.00 → 0.92; ambiguous 0.54 → 0.52) | Failed Recall (ambiguous 0.54 → 0.48) |
| Latency per call (p50 / p95) | | 171 / 230 ms | 453 / 1,109 ms | 1,590 / 3,223 ms |
| No response (of 3,600) | | 0 | 12 (five-second timeout) | 4 |
| Cost (3,600 calls) | | About $0.066 | $0.215 | $0.165 |

Jev was measured on the top 150 candidates (9,000 calls); `--top-k 60` selects the same candidate set for the comparison. On all 150 candidates, it achieved nDCG@10 of 0.912 (difference +0.159; 95% interval 0.090–0.234) at a cost of $0.164. The 3,600-call Jev cost is prorated from the 9,000-call measurement.

The context-inclusion threshold (`relevance_floor`) is calibrated per implementation. The highest threshold that preserves Recall was 0.25 for Jev (precision 0.12 → 0.52) and 0.40 for Clef (precision 0.11 → 0.53). For the Yes/No LLM, any threshold above zero merely excludes “No” results.

Because the LLM does not return probabilities, it can only move Yes candidates to the front while retaining their original order. Its inability to rank candidates with the same “Yes” response is considered the main reason it trails Jev.

## Run your own queries

```sh
uv sync --extra jev --extra llm
cp .env.example .env
# Configure credentials for the implementation you intend to run in .env
uv run --env-file .env python -m rerank_eval.run jev --output results/my-jev.json --max-usd 0.5
uv run --env-file .env python -m rerank_eval.run clef --top-k 60 --output results/my-clef.json --max-usd 1
uv run --env-file .env python -m rerank_eval.run llm --effort low --top-k 60 --output results/my-llm.json --max-usd 1
uv run python -m rerank_eval.evaluate results/my-jev.json
```

- Put API keys in `.env` at the repository root; create it from `.env.example` before first use. `.env` is excluded from Git, so real keys are not committed. Jev requires `TYPESAFE_API_KEY`; Clef requires `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`; the LLM requires `OPENAI_API_KEY`. In the Cloudflare dashboard, select the REST API under Workers AI to obtain a token and Account ID.
- Before this change, configuration files were not read: only shell environment variables were passed to the SDKs. For example, use `TYPESAFE_API_KEY=... uv run ... jev` or `OPENAI_API_KEY=... uv run ... llm` to set a key for one command. `AsyncTypeSafeClient` and `AsyncOpenAI` automatically read these standard environment variables. Clef gained direct-run support in this change and uses the Cloudflare Workers AI REST API.
- Before every call, the runner checks the remaining budget against a worst-case cost: the request’s UTF-8 byte count plus 1,024 tokens, the output limit. It stops before exceeding `--max-calls` or `--max-usd`.
- Each response is written to `<output>.jsonl` before the next request, so stopping partway through does not charge the same candidate twice.
- SDK automatic retries are disabled: one invocation equals one HTTP request. Failed calls are charged at the worst-case cost and retained as “no response” results.
- The included Clef results (`results/clef.json`) were obtained through the Clef adapter in Scaler production (Cloudflare Workers AI). `run clef` sends the same `state` and `noul` questions through the Cloudflare Workers AI REST API. As with the other results, scoring is reproducible without an API key.
- LLM requests use the same developer prompt, question, and strict JSON schema (`{"useful": boolean}`) as Scaler production’s LLM fallback. The included `gpt-6-luna-medium.json` was obtained through the production adapter.
- Prices use public rates as of 2026-10-07: Jev input $0.042 / 1M tokens, Clef input $0.24 / 1M tokens, and gpt-6-luna input $0.10 / output $0.50. Jev and Clef have no output charge; LLM reasoning tokens count as output.

## Scoring rules

- **Coverage:** A candidate covers an answer if it contains at least half of the characters in the answer’s quoted passage. Sections longer than 2,000 characters are chunked with a 200-character overlap.
- **nDCG@10:** Gain is `2^grade − 1`; discount is `log2(rank + 1)`. The ideal score is calculated from the complete candidate set.
- **Recall:** Grades 2 and above are relevant. Its denominator is fixed to the reachable relevant passages in the candidate set. Questions without answers are excluded rather than scored as perfect.
- **Reranking:** Descending probability (or 1/0 for Boolean responses), with original rank breaking ties. Candidates with no response are placed last in their original order.
- **TopK:** The number of answers per question in a result file defines that result’s candidate-set size. The nDCG ideal is always calculated from the full top 150.
- **Context selection:** Following the order, candidates with duplicate text or below the threshold are skipped; select at most 12, with at most four per document.

## Candidate-set construction and limitations

Production search was reproduced locally once and the results were frozen.

1. Split 41 documents into 221 chunks by heading (2,000 characters with 200-character overlap).
2. Embed chunks and search questions once with `text-embedding-3-small` (1,536 dimensions).
3. Rank all chunks by exact cosine similarity and save the top 150.

Follow-up questions that rely on a previous conversation were rewritten as standalone questions from the question and history alone (`search_query`).

Limitations:

- The corpus is synthetic English, so it lacks the diversity of real documents, such as tables and PDF extraction artifacts.
- Approximate nearest-neighbor search in production may swap closely ranked results. Search latency is not measured.
- Neither affects comparing two orderings of the same candidate set.
