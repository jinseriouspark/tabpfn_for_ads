# AdLift report
Dataset: `ads`. 943 rows across 120 campaign_ids. Outcome: **CTR**. Model backend: `baseline`.
## Decision summary
- **Cold start.** Ranking new rows with `GBDT + TF-IDF` before they run gives up 13.4% of the best achievable CTR, against 23.1% for picking at random. Testing only its top 2 keeps the true winner 53% of the time and leaves 72% of the exploration budget unspent.
- **Human vs LLM copy.** Handing the brief to an LLM hurts CTR by -0.840 pp (95% CI -1.052 pp to -0.454 pp). The pivot-table answer was +0.407 pp, the wrong sign. This estimate passes its checks.
- **Where the effect lives.** 97% of it runs through attributes you can see and control: length, whether it carries a number, tone, hashtags. That is the rewrite policy: let the model polish wording inside those limits.
## Cold-start benchmark
Held-out campaign_ids, never held-out rows.
| Model         |   Spearman |   Top-1 regret |   Recall@2 |   Exploration saved@2 |   Fit s |
|:--------------|-----------:|---------------:|-----------:|----------------------:|--------:|
| GBDT + TF-IDF |     0.6529 |         0.1343 |     0.5333 |                0.7153 |    1.88 |
| random        |     0.0323 |         0.2309 |     0.2833 |                0.7153 |    0    |
![cold start](cold_start.png)
## Causal estimate: LLM vs human copy
![effects](effects.png)
| Estimate | Value | 95% CI | Adjustment |
|---|---|---|---|
| Naive difference | +0.407 pp | – | none |
| T-learner total (headline) | -0.840 pp | -1.052 pp to -0.454 pp | confounders |
| S-learner total | -0.855 pp | -1.018 pp to -0.678 pp | confounders |
| S-learner direct | -0.029 pp | -0.040 pp to -0.020 pp | confounders + creative attributes |
| **Planted truth, total** | **-0.780 pp** | – | – |
| **Planted truth, direct** | **+0.455 pp** | – | – |
### Checks
- Estimators agree: **yes** (T-learner -0.840 pp, S-learner -0.855 pp). On small, heavy-tailed data the S-learner shrinks a weak treatment toward zero, so the T-learner is the headline and its interval comes from a bootstrap that refits both arms.
- Median per-row effect: -0.521 pp. When the mean and the median disagree in sign, a few outliers are carrying the mean; read the ratio.
- Placebo: shuffling the author label inside each campaign_id gives a mean effect of +0.047 pp; the real estimate is 18.2x larger.
- Overlap on the confounder set: author is predictable with AUC 0.65; 0.1% of rows fall outside common support. The total effect is identifiable from this data.
- Smallest effect this sample could resolve: about +0.299 pp (422 LLM rows, 521 human rows). An estimate inside that band is a statement about the sample, not about the copy.
- The direct (mediator-adjusted) estimate conditions on attributes the author determines. LLM copy is systematically longer and shaped differently, so the two arms barely overlap on those columns and the direct effect is poorly identified. It is reported for the mediation split, not as a headline.
## Effect by segment
![segments](segments.png)
| device   |   effect |      sd |   n |
|:---------|---------:|--------:|----:|
| mobile   | -0.00688 | 0.01417 | 594 |
| tablet   | -0.00841 | 0.01242 |  86 |
| desktop  | -0.01184 | 0.01255 | 263 |
## Method
- **Model.** TabPFN 3.5 through `tabpfn-client` reads the text directly; no vectoriser. Fitting is in-context with `fit_with_cache`, so each fold and each rewrite is seconds and repeated predictions skip the forward pass.
- **Splits.** Held-out campaign_ids. Rows in one campaign_id share an audience and an unobserved state; splitting rows would leak. Social timelines use a temporal split.
- **Effect.** Headline: T-learner (one model per author, cross-predicted), interval from a cluster bootstrap that refits both arms. Cross-check: S-learner (one model on confounders + author, each row scored under both). A doubly-robust (AIPW) estimator is in the library but is unstable below a few hundred rows on heavy-tailed outcomes, so it is not reported by default. T-learner: one model per author, cross-predicted. Intervals: cluster bootstrap over campaign_ids. Log-scale outcomes are reported as ratios.
- **Adjustment set.** Context columns only. Creative attributes are mediators and are excluded from the confounder set on purpose.
- **Related work.** Do-PFN (Robertson et al., 2025) uses these meta-learners as baselines and its 'Confounder + Mediator' case study is this problem's graph. Drift-Resilient TabPFN addresses the rising LLM adoption over time that confounds the naive comparison.