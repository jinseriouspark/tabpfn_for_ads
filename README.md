# AdLift — what makes a post land, before you post it. On TabPFN 3.5.

**A Threads account that started today has no history. AdLift scores a draft's expected views, says which levers (time, media, length, hook, hashtags, human vs LLM author) actually move reach on *this* account with honest intervals, rewrites the draft inside the account's own limits, and gets useful within the first few dozen posts. The same engine runs on ad creatives with click-through rate.**

TabPFN 3.5 is the reason the loop works at all. It fits in-context in seconds, reads the post text as text, returns a predictive distribution, and is built for the first fifty rows rather than the first fifty thousand. `fit_with_cache` keeps the account's context on the server so scoring a stack of rewrites costs one predict each.

| | |
|---|---|
| **Score before posting** | Predicted views with a band, plus the closest past posts and what they got. |
| **Levers, with intervals** | "Post as a reply: −62%. Evening instead of midday: +44%. Attach media: +38%." Each with a cluster-bootstrap CI and a support count. |
| **Rewrite inside the data** | The language model rewrites under limits read off the account's winners; TabPFN ranks the variants. |
| **Human vs LLM, causally** | Adoption timing and topic mix confound the pivot table. AdLift adjusts, cross-checks two estimators, runs a placebo, and reports the smallest effect the sample could have resolved. |
| **Learns from post #10** | A temporal learning curve shows where the ranking becomes trustworthy. |
| **Ships as** | CLI, Python library, MCP server, two cookbook notebooks. Runs offline with no keys; upgrades to hosted TabPFN 3.5 + Claude when keys are present. |

Hackathon categories: **Build an agent** (LLM proposes, TabPFN scores, the loop acts), **Build an extension or app** (MCP server + CLI), **Formalize a new problem** (social posts and ad creatives as a grouped, confounded, text-bearing tabular task with a known-truth generator).

---

## Quickstart (60 seconds, no keys)

```bash
git clone https://github.com/jinseriouspark/tabpfn_for_ads
cd tabpfn_for_ads
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev,mcp]"
source .venv/bin/activate

adlift demo --spec threads --out reports_threads   # virality engine on a synthetic timeline
adlift demo --spec ads     --out reports           # ad creatives with CTR
pytest -q                                         # 57 tests, offline
```

Score, suggest and rewrite a draft post:

```bash
adlift threads score   --text "Cut my support load 40% with one change: batching replies." --hour 20 --media
adlift threads suggest --text "Excited to share our journey! 🚀 #ai #startup #growth" --hour 9 --weekday Sat
adlift threads revise  --text "Excited to share our journey! 🚀 #ai #startup #growth" --n 6
adlift levers --spec threads
```

Hosted TabPFN 3.5 (recommended; this is what the project is built for):

```bash
uv pip install --python .venv/bin/python -e ".[client]"
export TABPFN_TOKEN=...          # https://ux.priorlabs.ai -> API key
adlift demo --spec threads       # adds the TabPFN 3.5 rows and learning curve next to the GBDT baseline
```

Claude for rewrites and banner reading:

```bash
uv pip install --python .venv/bin/python -e ".[llm]"
export ANTHROPIC_API_KEY=...
```

Without keys the model falls back to an untuned gradient-boosted baseline and the language model to a deterministic stub. Every output names the backend that produced it.

> **What is verified here and what is not.** The numbers in this README come from the offline baseline in the environment this was built in, whose network policy did not allow reaching `api.priorlabs.ai`. The hosted adapter follows the cookbook conventions (`create_default_for_version(ModelVersion.V3_5)`, `fit_mode="fit_with_cache"`, categorical indices, `group_col` in Thinking mode, explicit quantiles) and is exercised in tests up to the request, but it has not been run against the live service from this repository. Run `adlift demo` with a token to add the TabPFN 3.5 rows.

---

## Your own Threads account

Real view counts are available only for posts you own, so the engine is built around your timeline.

```bash
# 1. Pull posts + insights (Threads API user token with threads_basic + threads_manage_insights)
adlift threads fetch --token "$THREADS_ACCESS_TOKEN" --tz Asia/Seoul --out data/threads_posts.csv

# 2. Label which posts a model wrote (the API cannot tell you; nothing here guesses)
adlift threads template data/threads_posts.csv --out data/threads_labels.csv   # fill author = human|llm, topic
adlift threads label   data/threads_posts.csv data/threads_labels.csv --out data/threads_labeled.csv

# 3. Everything else takes --data
adlift demo   --spec threads --data data/threads_labeled.csv --out reports_mine
adlift levers --spec threads --data data/threads_labeled.csv
adlift threads suggest --data data/threads_labeled.csv --text "..."
```

Or assemble a CSV by hand with at least `text`, `timestamp`, `views` (optional: `likes`, `replies`, `reposts`, `quotes`, `media_type` or `has_media`, `is_reply`, `topic`, `author`). Posting hour, weekday, ISO week, week index and the text attributes are derived.

The API adapter (`src/adlift/ingest/threads.py`) was written against the documented v1.0 shapes (`/me/threads`, `/{id}/insights?metric=views,likes,replies,reposts,quotes`) but could not be exercised from the build environment; it prints the raw error body if a field or metric is rejected, so treat the first run as a check.

---

## What `adlift demo --spec threads` shows

The synthetic timeline is one account over 16 weeks with every lever's effect planted on the log of views, LLM adoption rising over time, a growing audience, heavy-tailed views with occasional viral runs, and **both potential texts** written for every post so the author effect is known exactly. Columns starting with `_truth_` are ground truth and never reach a model.

### Levers (relative to what the account usually does)

| lever | from → to | change in views | 95% CI | support |
|---|---|---:|---|---:|
| post as a reply | 0 → 1 | −62% | −70% to −55% | 32 |
| posting time | evening → midday | −44% | −48% to −40% | 25 |
| topic | productivity → personal | +53% | +45% to +60% | 23 |
| attach an image or video | 0 → 1 | +38% | +31% to +44% | 69 |
| day of week | weekday → weekend | −14% | −16% to −11% | 28 |
| hashtags | 0 → 4 | −8% | −11% to −7% | 52 |
| include a link | 0 → 1 | −5% | −7% to −4% | 34 |

![levers](reports_threads/levers.png)

`support` is how many posts were actually observed at the alternative. A lever with no support (the 45- and 70-word settings on a short-form account) is an extrapolation, and the table says so.

### Human vs LLM copy, on a heavy-tailed outcome

| Estimate | Value | 95% CI |
|---|---:|---|
| Naive difference (pivot table) | −46% | – |
| **T-learner, total effect (headline)** | **−42%** | **−49% to −16%** |
| S-learner, total effect | −32% | −41% to −22% |
| Planted truth, total | −22% | – |

The interval covers the truth; the point estimate does not sit on it. That is the honest state of a 171-post timeline with viral outliers: **the smallest author effect this sample can resolve is about ±28%**, and the report says so next to the number. The big levers above are estimated tightly because they have strong signals and balanced support; the author effect is modest and needs more posts. Across four generator seeds the refit-bootstrap interval covered the planted truth every time; the cheaper per-row bootstrap did not, which is why the headline interval refits both arms.

![effects](reports_threads/effects.png)

### Cold start and the learning curve

Temporal split: fit on the past, rank the next weeks' posts. GBDT + TF-IDF gives up 29% of the best achievable views on its first pick (random: 36%) and keeps the true winner in its top 2 a third of the time. The learning curve (`learning_curve.png`) is what a new account actually lives: ranking quality on the next ten posts as the context grows from ten. With a token, the TabPFN 3.5 line is drawn next to it; the whole argument for TabPFN is what happens on the left of that chart.

![learning curve](reports_threads/learning_curve.png)

## What `adlift demo --spec ads` shows

The ad account (943 creatives, 120 campaigns) is the same construction with CTR as the outcome and campaign as the group.

| Model | Spearman | Top-1 regret | Recall@2 | Exploration saved@2 |
|---|---:|---:|---:|---:|
| GBDT + TF-IDF | 0.65 | 13.4% | 53% | 72% |
| random | 0.03 | 23.1% | 28% | 72% |

| Estimate | Value | 95% CI |
|---|---:|---|
| Naive difference | **+0.41 pp** | – |
| **T-learner, total effect (headline)** | **−0.84 pp** | **−1.05 to −0.45** |
| S-learner, total effect | −0.86 pp | −1.02 to −0.68 |
| Planted truth, total | **−0.78 pp** | – |

The pivot table gets the sign wrong: LLM copy was adopted later and in better verticals. The adjusted estimate recovers the planted truth; placebo 18×, overlap AUC 0.65 with 0.1% of rows off support. 97% of the effect runs through visible attributes (length, numbers, tone), which is the rewrite policy.

![effects](reports/effects.png)

---

## MCP server

```bash
uv pip install --python .venv/bin/python -e ".[mcp]"
adlift-mcp                         # stdio; or: python -m mcp_server.server
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "adlift": {
      "command": "/absolute/path/to/tabpfn_for_ads/.venv/bin/adlift-mcp",
      "env": { "TABPFN_TOKEN": "...", "ANTHROPIC_API_KEY": "...", "ADLIFT_BACKEND": "auto" }
    }
  }
}
```

Claude Code: `claude mcp add adlift -- /absolute/path/to/tabpfn_for_ads/.venv/bin/adlift-mcp`

| Tool | What it does |
|---|---|
| `load_threads(csv_path="", access_token="", outcome="views")` | Load a labelled timeline, pull one through the API, or use the synthetic one; fit in seconds. |
| `score_post(text, posted_hour, weekday, has_media, is_reply, topic, author)` | Predicted views with a band and the closest past posts. |
| `suggest_post(text, …)` | Which single change (time, media, length, hook, hashtags, author) raises this post most. |
| `revise_post(text, …, n_variants)` | LLM rewrites inside the account's limits, ranked by predicted views with lift. |
| `lever_report()` | Every lever's effect vs. the account's habits, with intervals and support. |
| `learning_curve_report()` | Ranking quality on the next block as the context grows. |
| `causal_report()` | Human-vs-LLM effect with T/S-learners, refit bootstrap, placebo, overlap, detectable effect. |
| `cold_start_benchmark()` | Regret, recall and budget saved on held-out groups. |
| `load_account`, `score_creative`, `revise_creative`, `ingest_banner` | The same for ad creatives; `ingest_banner` reads a banner image with a vision model. |
| `account_status()` | What is loaded, which backends and keys are active. |

Every tool returns JSON and names its backend.

---

## Method

- **Model.** `CTRModel` wraps hosted TabPFN 3.5 (`tabpfn-client`, text passed as text, `fit_with_cache`, explicit quantiles), local open-source `tabpfn`, and an untuned `HistGradientBoostingRegressor` over TF-IDF→SVD + one-hot as baseline and offline fallback. Outcomes are modelled on their natural scale (CTR) or `log1p` (views) and reported as points, ratios and bands.
- **Column roles** (`DatasetSpec`): text; creative attributes (mediators: the author determines them); context (confounders: decided before writing); treatment; group; outcome. `AD_SPEC` and `THREADS_SPEC` ship; a CSV with the same roles can declare its own.
- **Cold start.** Held-out groups, never held-out rows: random over campaigns for ads, temporal for posts. Metrics are computed the way the budget is spent (top-1 regret, recall@k, exploration saved) plus a learning curve in time order.
- **Levers.** One model on the structured attributes (no raw text); every row re-scored under each alternative setting of one lever; cluster bootstrap over groups; support counts.
- **Author effect.** T-learner headline with a cluster bootstrap that refits both arms; S-learner cross-check; within-group placebo; propensity overlap; mediation split; smallest resolvable effect. Adjustment set is context only. A doubly-robust (AIPW) estimator is in the library but its cross-fitted score is unstable below a few hundred rows on heavy-tailed outcomes, so it is not in the default report.
- **Rewrite loop.** `CopyCoach` fits once, derives `RewriteConstraints` from the account's top quartile, asks the language model for N variants, scores draft and variants in one call, and returns them ranked with lift and band; `precedents` shows the closest past posts.

### Related Prior Labs work

- **Do-PFN** ([Robertson et al., 2025](https://arxiv.org/abs/2506.06039)): its baselines are the S- and T-learners used here and its "Confounder + Mediator" case study is this problem's graph. `adlift analyze --dopfn` runs it side by side from a clone of [jr2021/Do-PFN](https://github.com/jr2021/Do-PFN) (`ADLIFT_DOPFN_PATH`).
- **Drift-Resilient TabPFN** ([arXiv 2411.10634](https://arxiv.org/abs/2411.10634)): rising LLM adoption over `week_index` is exactly the temporal shift that confounds the naive comparison.
- **FairPFN** ([arXiv 2407.05732](https://arxiv.org/abs/2407.05732)): the counterfactual member of the family; "what if the other author had written it" is the same shape.
- Changelog features used: KV cache (`fit_with_cache`) for the scoring loop; the decoder-readout cookbook is the model-native version of `precedents` (local model only, so the hosted path uses similarity).
- Cookbook recipes built on: *Predict Restaurant Ratings with TabPFN 3.5*, *Experiment with Thinking Mode*, *Get Started with Predictive Distribution*, *Faster Inference with KV Cache*.

---

## Layout

```
src/adlift/
  schema.py          DatasetSpec / OutcomeSpec; AD_SPEC and THREADS_SPEC; AdCreative
  text_features.py   attributes the author controls (length, number, question, hashtags, link, emoji, CTA)
  datasets/          synth.py (ads) and synth_threads.py (one account), both with known truths
  model.py           CTRModel: client | local | baseline, outcome transforms, KV cache
  coldstart.py       grouped / temporal cold start, learning curve
  causal.py          T/S/DR learners, refit cluster bootstrap, placebo, overlap, mediation, Do-PFN hook
  levers.py          per-lever counterfactuals and per-post suggestions
  llm.py             Claude + stub: banner extraction, constrained rewrites (ad or post)
  loop.py            CopyCoach: fit once, score / revise / precedents; PostDraft
  ingest/            threads.py (API + CSV), banner.py, adapter.py (ads CSV)
  report.py          figures (validated colour-blind-safe palette) + Markdown report
  cli.py             adlift synth | benchmark | curve | analyze | levers | demo | score | revise | ingest | threads ...
mcp_server/server.py     MCP server (stdio)
notebooks/               adlift_cookbook.ipynb (ads), threads_virality_engine.ipynb (Threads)
tests/                   57 tests, offline
reports/, reports_threads/   output of the two demos
```

## License

MIT.
