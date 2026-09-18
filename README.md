# AdLift — real-time causal copy testing on TabPFN 3.5

**Predict how a new ad creative will perform before it serves a single impression. Then answer the question every marketing team is asking: does LLM-written copy actually work better than human copy, and if not, how should a human revise with AI help?**

AdLift puts TabPFN 3.5 in the loop between a language model that *writes* ad copy and the budget decision that *funds* it. TabPFN fits in-context on an advertiser's history in seconds, so a brand-new creative gets a click-through-rate prediction with an uncertainty band instantly, and a stack of LLM-generated rewrites gets ranked at the cost of one `predict` call. A causal layer on the same model separates what LLM copy *does* from what it merely *correlates with*.

| | |
|---|---|
| **Cold start** | Rank creatives in campaigns the model has never seen. Cuts exploration spend by 72% while keeping the true winner in the top 2 more than half the time. |
| **Causal answer** | On the reference account the pivot table says LLM copy is +0.41 pp better. The adjusted estimate says −0.86 pp. The planted truth is −0.78 pp. Naive got the sign wrong. |
| **Rewrite policy** | 97% of the LLM penalty runs through three visible habits: longer copy, dropped numbers, aspirational tone. Constrain those and let the model polish. |
| **Ships as** | CLI, Python library, MCP server, cookbook notebook. Runs offline with no keys; upgrades itself to hosted TabPFN 3.5 + Claude when keys are present. |

Hackathon categories: **Build an extension or app** (MCP server + CLI), **Formalize a new problem** (ad creatives with free text as a grouped, confounded tabular task), **Build an agent** (LLM proposes, TabPFN scores, loop acts).

---

## Quickstart (60 seconds, no keys required)

```bash
git clone https://github.com/jinseriouspark/tabpfn_for_ads
cd tabpfn_for_ads
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev,mcp]"
source .venv/bin/activate

adlift demo            # cold-start benchmark + causal report -> reports/report.md
pytest -q              # 43 tests, ~40 s, no network
```

Add hosted TabPFN 3.5 (recommended; this is what the project is built for):

```bash
uv pip install --python .venv/bin/python -e ".[client]"
export TABPFN_TOKEN=...          # https://ux.priorlabs.ai -> API key
adlift demo                      # the TabPFN 3.5 row appears next to the GBDT baseline
```

Add Claude for banner reading and copy rewriting:

```bash
uv pip install --python .venv/bin/python -e ".[llm]"
export ANTHROPIC_API_KEY=...
adlift revise --headline "Discover how Atlas can transform the way you handle ticket backlogs."
adlift ingest examples/banner_demo.jpg --device mobile --placement social_feed
```

Without keys every command still runs: the model falls back to an untuned gradient-boosted baseline and the language model to a deterministic stub. Every output says which backend produced it.

---

## What you get from `adlift demo`

The reference account is synthetic on purpose. Real creative-level ad data with copy attached is proprietary, and a causal method that cannot be checked against a known answer is not worth trusting. The generator writes **both potential copies** for every creative (human-style and LLM-style), computes the click-through rate under each, and hides the truth in `_truth_*` columns the model never sees. It also plants the three things that make the real problem hard: confounding (LLM adoption rises over time and clusters by vertical), mediation (the author changes copy length, tone and numeric claims), and grouping (creatives nest in campaigns that share a budget and an unobserved quality).

Numbers below are from the offline baseline backend in this repository's CI environment. Run with `TABPFN_TOKEN` to add the TabPFN 3.5 row.

### Cold-start benchmark (held-out campaigns, `GroupKFold`)

| Model | Spearman | Top-1 regret | Recall@2 | Exploration saved@2 | Fit (s) |
|---|---:|---:|---:|---:|---:|
| GBDT + TF-IDF | 0.655 | 13.1% | 54% | 72% | 1.7 |
| random | 0.032 | 23.1% | 28% | 72% | 0 |

*Top-1 regret*: CTR given up by scaling the model's first pick instead of the true winner. *Recall@2*: share of new campaigns whose true best creative is in the model's top two. *Exploration saved@2*: exploration budget left unspent if only the top two are tested.

![cold start](reports/cold_start.png)

### Does LLM copy outperform human copy?

| Estimate | Value | 95% CI | Adjustment |
|---|---:|---:|---|
| Naive difference (pivot table) | **+0.407 pp** | – | none |
| S-learner, total effect | **−0.855 pp** | −1.018 to −0.678 | confounders |
| T-learner, total effect | −0.840 pp | −1.086 to −0.623 | confounders |
| S-learner, direct effect | −0.029 pp | −0.040 to −0.020 | confounders + creative attributes |
| Planted truth, total | **−0.780 pp** | – | – |
| Planted truth, direct | +0.455 pp | – | – |

![effects](reports/effects.png)

Checks, all reported alongside the number:

- **Two estimators agree** (S-learner −0.855, T-learner −0.840).
- **Placebo**: shuffling the author label within campaigns gives +0.047 pp; the real estimate is 18× larger.
- **Overlap**: on the confounder set, author is predictable with AUC 0.65 and 0.1% of rows sit off common support. The total effect is identifiable.
- **The direct effect is not.** Adjusting for creative attributes gives AUC 0.997 and 57% of rows off support: LLM copy is systematically longer and less numeric, so there is almost no human copy that looks like LLM copy to compare against. The direct estimate is shown for the mediation split, not as a headline. This is exactly the "Confounder + Mediator" case study in the Do-PFN paper where meta-learners are expected to struggle.

Effect by device (the average hides a spread):

![segments](reports/segments.png)

---

## The questions this answers

**Which is better, human or machine text?** On this account, human. Handing the brief to an LLM lowers CTR by about 0.8 points, and the naive comparison says the opposite because LLM copy was adopted later and in better-performing verticals. On *your* account the answer may differ; the point of the tool is that it gives you the adjusted answer for your data, with its checks.

**Why?** Not because of the words the model chooses. The direct effect of LLM authorship, holding the copy's shape fixed, is small and poorly identified. 97% of the total effect travels through three attributes you can see: LLM copy runs longer, drops concrete numbers, and reaches for aspirational verbs. Those attributes have their own, negative, effect on clicks.

**How should a human revise with AI help?** Keep the shape, let the model polish the wording. `adlift revise` derives the limits from what has won on this account (max words, keep the number, preferred tone) and asks the language model to rewrite inside them, then TabPFN ranks the results:

```
$ adlift revise --headline "Discover how Northwind can transform the way you handle manual reviews." \
                --body "Join thousands of forward-thinking teams." --cta "Begin your journey"

Draft 3.80% -> best 4.82%  (fit 2.6s, score 0.02s)
  #   pred CTR   lift  words  copy
  draft  3.80%           17  Discover how Northwind can transform ... / Begin your journey
  1      4.82%  +1.02     6  Cut manual reviews. / No setup. Cancel anytime. / Begin your journey
  2      4.70%  +0.90     4  Fix manual reviews. / Begin your journey
constraints: <= 21 words, tone direct, keep number: True
```

**Can this run in real time, from a cold start?** Yes, and that is the reason TabPFN is the right model. Fitting is in-context: the advertiser's history becomes context, no weights change, no hyperparameters are searched. A new advertiser with 300 creatives gets a usable ranker in seconds. Scoring six rewrites is one `predict` call (0.02 s above). With hosted TabPFN the `predict_interval` band is the model's own predictive distribution, not a residual heuristic.

**Is it meaningful?** Yes, with two conditions the tool enforces. First, the estimate has to survive the placebo and overlap checks, which are always printed next to it. Second, it is a decision aid for *which creative to fund and how to rewrite*, not a proof about LLMs in general. The synthetic account demonstrates that the machinery recovers a planted truth the pivot table gets backwards; your data decides the sign.

---

## MCP server

AdLift ships as a Model Context Protocol server so an agent or an editor can score, rewrite and audit creatives through tool calls.

**Run it**

```bash
uv pip install --python .venv/bin/python -e ".[mcp]"
adlift-mcp                           # stdio transport
# or: python -m mcp_server.server
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "adlift": {
      "command": "/absolute/path/to/tabpfn_for_ads/.venv/bin/adlift-mcp",
      "env": {
        "TABPFN_TOKEN": "...",
        "ANTHROPIC_API_KEY": "...",
        "ADLIFT_BACKEND": "auto"
      }
    }
  }
}
```

**Claude Code**

```bash
claude mcp add adlift -- /absolute/path/to/tabpfn_for_ads/.venv/bin/adlift-mcp
```

**Tools**

| Tool | What it does |
|---|---|
| `load_account(csv_path="", n_campaigns=120, seed=7)` | Load a CSV in the AdLift schema, or the synthetic account, and fit in seconds. |
| `score_creative(headline, body, cta_text, author, device, placement, …)` | Predicted CTR with an uncertainty band for a creative that has never served. |
| `revise_creative(headline, …, n_variants=6, max_words=0, tone="")` | LLM rewrites under data-derived limits, ranked by TabPFN, each with its lift over the draft. |
| `ingest_banner(image_path_or_url, device, placement, …)` | Read a banner image into the schema with a vision model, optionally score it. |
| `causal_report(n_boot=200, segment_by="device")` | S/T-learner effect of LLM authorship with CI, placebo, overlap, mediation, segments. |
| `cold_start_benchmark(n_splits=4)` | Regret, recall and budget saved on held-out campaigns. |
| `account_status()` | What is loaded and which backends and keys are active. |

Every tool returns JSON. Without keys the server runs on the baseline model and stub language model and says so in every response.

---

## Use your own data

`adlift` reads a CSV with one row per creative. Required: `headline`, `campaign_id`, `author` (`human` or `llm`), and either `ctr` or both `clicks` and `impressions`. Everything else is optional and defaults sensibly.

```bash
adlift demo --data my_account.csv --out reports/
adlift analyze --data my_account.csv --segment-by placement
adlift score --data my_account.csv --headline "..." --device mobile
```

Column groups (see `src/adlift/schema.py`):

| Group | Columns | Role in the model |
|---|---|---|
| Text | `headline`, `body`, `cta_text` | Read directly by TabPFN 3.5 |
| Creative attributes | `word_count`, `char_count`, `has_numeric_claim`, `claim_type`, `tone`, `has_brand_logo`, `has_human_face`, `background_kind`, `dominant_color`, `text_contrast` | Features for prediction; **mediators**, excluded from the causal adjustment set |
| Context | `vertical`, `placement`, `device`, `audience`, `daily_budget_usd`, `week_index` | Features and **confounders** |
| Treatment | `author` | The thing whose effect is estimated |
| Group | `campaign_id` | Splits, `group_col`, cluster bootstrap |

Rename on the way in with `frame_from_csv(path, column_map={"title": "headline", ...})`. Validation refuses data where every campaign has a single author, because that makes the author perfectly confounded with the campaign.

---

## Method

- **Model.** `CTRModel` wraps three backends behind one `fit` / `predict` / `predict_interval`: hosted TabPFN 3.5 via `tabpfn-client` (text columns passed as text, `group_col=campaign_id`, `output_type="full"` for the distribution), local open-source `tabpfn`, and an untuned `HistGradientBoostingRegressor` over TF-IDF→SVD plus one-hot as the baseline and offline fallback.
- **Cold start.** `GroupKFold` on campaign. Metrics are computed per held-out campaign the way the budget is spent: regret of the top pick, recall of the true winner at k, share of exploration budget unspent.
- **Causal effect.** S-learner (one model on confounders + author, each row scored under both authors) and T-learner (one model per author, cross-predicted). Cluster bootstrap over campaigns for intervals. Placebo by within-campaign label permutation. Propensity-based overlap diagnostic. Mediation split = total − direct.
- **Adjustment set.** Context columns only. Creative attributes sit on the causal path from author to outcome; conditioning on them removes part of the effect being measured. Adjusting for the text itself would condition on the treatment's own output.
- **Rewrite loop.** `CopyCoach` fits once, derives `RewriteConstraints` from the account's top-quartile creatives, asks the language model for N variants inside those limits, scores draft and variants in one call, and returns them ranked with lift and band.

### Related Prior Labs research

- **Do-PFN** ([Robertson et al., 2025](https://arxiv.org/abs/2506.06039)) estimates interventional outcomes directly from observational data. Its baselines are the S- and T-learners used here, and its "Confounder + Mediator" case study is this problem's graph. `adlift analyze --dopfn` runs Do-PFN side by side when `ADLIFT_DOPFN_PATH` points at a clone of [jr2021/Do-PFN](https://github.com/jr2021/Do-PFN).
- **Drift-Resilient TabPFN** ([arXiv 2411.10634](https://arxiv.org/abs/2411.10634)) targets temporal distribution shift. LLM adoption rising over `week_index` is exactly that shift, and is the main confounder here.
- **FairPFN** ([arXiv 2407.05732](https://arxiv.org/abs/2407.05732)) is the counterfactual member of the family; the "what if the other author had written it" question is the same shape.
- Cookbook recipes this builds on: *Predict Restaurant Ratings with TabPFN 3.5* (text + tabular), *Experiment with Thinking Mode* (`group_col`), *Get Started with Predictive Distribution* (`output_type="full"`), *Faster Inference with KV Cache*.

---

## Layout

```
src/adlift/
  schema.py        canonical creative table; column roles (text / mediator / confounder / treatment / group)
  datasets/synth.py synthetic account with both potential outcomes and a planted effect
  model.py         CTRModel: client | local | baseline backends
  coldstart.py     grouped cold-start evaluation: regret, recall@k, exploration saved
  causal.py        S/T-learner ATE, cluster bootstrap, placebo, overlap, mediation, optional Do-PFN
  llm.py           Claude + stub backends: banner extraction, constrained rewrites
  loop.py          CopyCoach: fit once, score/revise instantly
  ingest/          banner image -> creative; CSV adapter
  report.py        figures (validated colour-blind-safe palette) + Markdown report
  cli.py           adlift synth | benchmark | analyze | demo | score | revise | ingest
mcp_server/server.py   MCP server (stdio)
notebooks/adlift_cookbook.ipynb  cookbook-style walkthrough
tests/             43 tests, offline
reports/           output of `adlift demo`
```

## License

MIT.
