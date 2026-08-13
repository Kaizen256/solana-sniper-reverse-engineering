# Six Seconds to Decide: Reconstructing a Solana Zero-Block Sniper

## 1. Problem and decision clock

We reconstruct wallet `5brv7...LyAr`, predict its deployment-time selections among 5,076,421 Pump.fun tokens, and test a replica without allowing future information into entry decisions. `t_decision` is deployment. Features use only the signed deployment message and deployer facts observed strictly before `deployment.blockTime`. Same-second history, transaction `meta`, target reactions, landed Jito facts, trades, candles, and future outcomes are excluded.

Historical outcomes require a second guard. A creator-fee claim or deployer sale on a prior launch enters history only at its observed timestamp; one- and seven-day denominators activate only after maturity. June outcomes are joined only after predictions freeze.

## 2. The bot's fast trading behavior

The core positive class contains 15,927 deployment tokens, all linked to a target first buy. The broader wallet record contains 16,163 bought positions and 87,007 activity rows.

| Requested statistic | Result |
|---|---:|
| Entry size, USD | mean **$263.74**; median **$184.12**; SD **$233.10**; IQR $136.03 |
| Entry size, SOL-equivalent quote | mean 8.678; median **2.469**; SD 55.542; IQR 1.481 |
| Deployment --> first buy | median **0 seconds / 0 slots**; p90 0 seconds / 1 slot |
| Same-slot entries | **12,683 / 15,927 (79.63%)** |
| Same-slot landed position | median **+118 transactions** after deployment; only 1 was literally next |
| Hold time | median **6 s**; IQR 5 s; p90 17 s; mean 10.62 s |
| Partial exits | **15,612 / 16,163 (96.59%)** |
| Sell structure | median **4 sells/token**; **70,805** sells across bought positions |
| Burns | **2 transactions / 2 tokens** |
| Net cash-flow hit rate | **64.25%** |
| Average winner / loser | **+$116.01 / -$27.06** |
| Activity-table cash flow | $4.255M bought; $5.743M sold; $439,840 costs; **+$1.048M net** |

The latency mean is 36.4 seconds because of a tiny extreme tail; medians and quantiles describe the fast path better. Transaction indexes establish landed order, not wall-clock latency, mempool visibility, or private ordering. Activity-table cash flow is not a complete marked residual-inventory ledger.

## 3. Leakage-safe store and preserved control

We streamed both deployment archives and decoded only signed-message facts: metadata, dev-buy arguments, ComputeBudget choices, account/instruction structure, and top-level System transfers. No `meta` field is read. Coverage is 5,070,147 deployments; 6,274 unsupported messages receive a missing flag.

For history, 177.4M activity rows become 158.4M wallet-second states. Every candidate ASOF-joins only a state with `activity.timestamp < deployment.blockTime`, plus strict 1/7/30-day differences. The store has 5,076,421 unique tokens, 15,927 labels, no duplicate keys, and no non-positive recencies. Provider histories are capped at 10,000 events per wallet, so age and activity totals are explicitly left-censored.

The preserved control is regularized LightGBM trained from the target's March 12 active start through April, with deterministic half-negative sampling and inverse weights. Model family and the 0.10347 threshold are selected on full May.

| Preserved control | Prevalence | PR-AUC | Precision | Recall | F1 | Entries |
|---|---:|---:|---:|---:|---:|---:|
| May selection | 0.570% | 0.14451 | 21.85% | 26.41% | 0.23917 | 6,137 |
| June reporting | 0.492% | 0.06872 | 14.45% | 13.44% | 0.13928 | 3,904 |

Thus June PR-AUC is 13.96x prevalence and precision is 29.34x prevalence-meaningful enrichment, but still low absolute precision under extreme imbalance. Accuracy is not reported.

## 4. Final imitation selector: strict-as-of prior-launch quality

Full historical token ROI, peak, migration, survival, and rug labels are absent. The defensible local proxies are a realized Pump creator-fee claim and the deployer's first observed sale of its own earlier token. These are timestamped economic/behavioral outcomes, not true token ROI, migration, survival, rug status, or profit.

Mechanical lifecycles update launches, claims, claim value, first developer sells, maturity, recency, and consistency only when observable. The 5,076,421-row audits find zero future states/launches, same-second outcome recencies, invalid fractions, or duplicate keys. If the provider cap retains a claim after its older launch falls outside the snapshot, ratios are bounded and an incomplete-history flag is set.

| Chronological window | Prevalence | Control PR-AUC | Final PR-AUC | Final precision | Recall | F1 | Entries |
|---|---:|---:|---:|---:|---:|---:|---:|
| March 12–31 --> April | 0.424% | 0.09705 | **0.10756** | 16.34% | 32.26% | 0.21696 | 6,730 |
| March 12–April --> May | 0.570% | 0.14465 | **0.15337** | 21.64% | 29.70% | **0.25037** | 6,969 |
| June reporting | 0.492% | 0.06872 | **0.06542** | 12.76% | 13.64% | 0.13181 | 4,484 |

Creator fees improve April/May PR-AUC by +0.01025/+0.00331; developer sells then add +0.00026/+0.00541. Their incremental daily delta is positive on 18/30 and 21/31 days. The final May threshold is 0.10033. In June, final PR-AUC is 13.29x prevalence and precision 25.91x prevalence, but both are slightly worse than the preserved control. We retain the augmented selector because every third-pass KEEP/DROP decision used pre-June chronological evidence.

June is a reporting test, not a pristine blind holdout: an early HGB run had already inspected it before LightGBM was introduced. No later choice uses June to tune this result.

## 5. Explicit Top-10 reverse-engineered features

This is the promoted selector's deterministic permutation ranking on a 200,000-row May sample (sample baseline PR-AUC 0.14013). Drops are single-feature perturbations and are not additive when predictors correlate. Effect patterns come from the frozen selector's 300,000-row May effect table.

| Rank | Feature | PR-AUC drop | Supported effect / pattern | Stability or evidence | Interpretation |
|---:|---|---:|---|---|---|
| 1 | `seconds_since_prior_deploy` | 0.03692 | Increasing; buy rate rises sharply beyond ~184 s and again beyond ~829 s | Control top-2 in April/May; spacing stays stable weekly | Rejects near-simultaneous launchers; favors meaningful spacing |
| 2 | `dev_buy_sol` | 0.02848 | Stronger above ~1.1 SOL; low/micro buys are weak | Control top-4 May/top-1 April; effect weakens in June | Signed dev commitment, but regime-dependent |
| 3 | `hist_quote_sol_sum` | 0.02466 | Higher at large historical SOL scale, especially the upper tail | Control top-6 April/top-3 May; redundant-group audit | Wallet scale/sophistication proxy |
| 4 | `hist_open_close_count` | 0.02286 | Mixed middle; high extreme (>~3,437) is elevated | Control top-4 April/top-2 May; redundant-group audit | Breadth/scale of prior activity |
| 5 | `hist_cost_usd_sum` | 0.02086 | Upper-tail historical USD value is elevated | Control top-5 in both windows; redundant-group audit | Fiat-valued manifestation of wallet scale |
| 6 | `latest_prior_launch_dev_sell_latency_seconds` | 0.01319 | Decreasing; 1–6 s prior sells score above >10 s | #1 developer-sell split feature in April and May; family adds +0.00541 May PR-AUC | Captures a repeatable fast-exit launch style |
| 7 | `prior_deploy_count_1d` | 0.00954 | Peaks around 2–5; falls sharply above ~39/day | Control top-3 April/top-9 May | Moderate cadence favored; industrial cadence rejected |
| 8 | `hist_burn_count` | 0.00901 | >6 historical events is elevated | Control top-7 April; correlated activity family | Another wallet-scale/activity manifestation, not a causal burn rule |
| 9 | `hist_burn_count_30d` | 0.00815 | >2 recent events is elevated | Final-selector May permutation; companion to cumulative activity | Recent scale/engagement manifestation |
| 10 | `hist_claim_fee_usd_per_claimed_launch` | 0.00801 | Sparse/non-monotonic alone; useful jointly | Creator-fee split top-4 April/top-5 May; family improves both windows | Realized fee quality per prior claimed launch |

Several historical-activity variables-SOL/USD value, open/close events, sells, tips, and burns are highly correlated manifestations of a latent wallet-scale/sophistication factor and must not be read as independent causal rules. Median pairwise Spearman among the audited core activity group is 0.941; one rank component explains 91.5%.

The separate rule hierarchy is: **strong** meaningful spacing, moderate cadence, signed dev-buy commitment, and the latent wallet-scale factor; **strong incremental** strict-as-of creator-fee realization and prior developer-sell timing; **plausible** observed wallet age, subject to left-censoring; **speculative/unsupported** counterparty graphs and wrapper internals, which are excluded.

## 6. Regime shift and the limit of one imitation rule

May and June target rates overlap (463–681 versus 412–706 buys per 100,000), so frequency drift is insufficient. Positive zero-dev-buy share changes from roughly 0–1% through April to 41.31% late May and 42.89–55.60% in June. Spacing remains comparatively stable, while dev-buy and wallet-scale effects weaken and signer composition changes. This is consistent with compositional/effect drift; it is not evidence of one abrupt bot rewrite. It also explains why outcome augmentation improves both pre-June windows yet transfers adversely to June.

Six-hour LambdaRank, 3x chronological hard negatives, a missing-history submodel, and expanded metadata fail their pre-June gates and remain dropped.

## 7. Imitation is not economic selection

Model A estimates target-buy probability. Model B separately estimates a creator-fee claim within seven days using only labels matured before evaluation. Model-B PR-AUC is 0.10450 in April (26.5x prevalence) and 0.05911 in May (14.2x). At equal Model-A trade count, two-stage reranking raises claim hit rate from 4.71%-->6.36% in April and 4.38%-->5.22% in May. The pre-June-selected 0.25x point is the final economic strategy: 1,121 June entries, 177 target overlaps, 15.79% overlap precision, and 4.22% target recall.

That selectivity is an economic choice, not a classification improvement. The selective policy is not a better imitation model simply because it trades fewer tokens.

## 8. Target versus replica, under the same execution diagnostic

| Cohort/policy | Entries | Overlap | Precision | Recall | Immediate hit / median ROI | +118 fill | +118 hit / median ROI | +118 p99-cap P&L | Drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Target cohort, equal-stake replay | 4,195 | - | - | - | 64.10% / +15.66% | 73.30% | 40.07% / -7.77% | +108.84 SOL | 58.12 SOL |
| Preserved control imitation | 3,904 | 564 | 14.45% | 13.44% | 62.98% / +15.29% | 62.76% | 43.22% / -4.88% | +172.77 SOL | 28.40 SOL |
| **Final selective two-stage** | **1,121** | **177** | **15.79%** | 4.22% | 62.01% / +14.39% | 64.41% | **48.61% / -2.06%** | **+215.76 SOL** | **11.41 SOL** |

These are comparable marginal-price diagnostics with the same 1.9753 SOL stake, six-second exit, 0.09101 SOL cost, and p99 cap, not the target's actual fills. Immediate execution is an optimistic bound. +118 is approximately the target's median landed-position offset; it is not 118 milliseconds or evidence of mempool visibility. Higher offsets change which tokens fill, so conditional ROI need not decline monotonically.

Separately, the target's **actual June activity-table cash flow** is +$213,884, 63.36% hit rate, $9.32 median net P&L, and 19.86% net ROI on buys plus recorded fees. That differently sized, multi-exit cash-flow measurement is not an exact-curve counterfactual and is not substituted into the equal-stake table.

## 9. Exact Pump mechanics and bounded conclusions

Integer constant-product replay materially reduces the optimism of marginal event prices by inserting our buy and sell, price impact, 0.95–1.25% swap fees, and intervening events. A targeted 169 MB raw batch decodes 82 exact current-cohort events, confirms fixed-quote and fixed-token buy variants plus 0/30 bps creator fees, and matches every normalized SOL/token amount.

At +118 and 0.95% fees:

| Strategy | Supported coverage | Median ROI bound | p99-capped P&L bound |
|---|---:|---:|---:|
| Preserved control | 46.90–48.72% | -9.44% to -7.54% | -117.5 to +9.7 SOL |
| Final imitation | 49.98–51.38% | -9.85% to -7.73% | -175.3 to -5.5 SOL |
| Equal-count two-stage | 51.58–52.74% | -9.04% to -6.93% | -155.5 to +28.0 SOL |
| **Selective two-stage** | **50.58–51.56%** | **-7.14% to -6.34%** | **+9.8 to +51.1 SOL** |

The selective strategy remains positive on capped P&L under both intent bounds (+2.8 to +43.9 SOL even at 1.25% fees), but its median is negative. Its advantage is principally selectivity and tail robustness-not robust positive median profitability, and not proof that it beats the bot.

Exact delayed counterfactual execution remains underidentified because the normalized table omits downstream instruction intent, slippage limits, and exact reserves. Unsupported nonstandard/Mayhem curves, PumpSwap migrations, counterfactual curve completion, and possible slippage failures are excluded rather than assigned fabricated prices. A target exact-curve replay of its actual variable sizing and multi-leg exits is therefore not claimed.

The most credible remaining improvement requires new timestamped pre-June market outcomes or richer raw/live execution data. Every reported number maps to tracked tables under `submission/tables/`; no live trading or external execution was performed.
