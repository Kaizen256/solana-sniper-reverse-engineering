# Six Seconds to Decide: Reconstructing a Solana Zero-Block Sniper

## 1. Problem and decision clock

The goal is to reconstruct wallet 5brv7...LyAr, predict which tokens it would buy at deployment time from a pool of 5,076,421 Pump.fun tokens, and separately evaluate whether those predictions could form a profitable trading strategy.

`t_decision` is the token deployment time. For information about a previous target buy to be available, it must satisfy:

`target_buy_time < current candidate block_time`

Equality is excluded. The target wallet's reaction to the current token, along with every action it takes afterward, is used only for labels and evaluation.

At prediction time, the model can use the signed deployment message and information about the deployer that was available strictly before deployment. Transaction meta, trades, candles, and landed Jito information are not used as features.

## 2. Fast behavior and a venue-aware fee ledger

The core positive class contains 15,927 deployed tokens that were followed by a target first buy. The broader wallet history contains 16,163 bought positions and 87,007 activity rows.

| Statistic | Result |
|---|---:|
| Entry size, USD | mean **$263.74**; median **$184.12**; SD **$233.10**; IQR $136.03 |
| Entry size, SOL-equivalent quote | mean 8.678; median **2.469**; SD 55.542; IQR 1.481 |
| Deployment → first buy | median **0 seconds / 0 slots**; p90 0 seconds / 1 slot |
| Same-slot entries | **12,683 / 15,927 (79.63%)** |
| Same-slot landed position | median **+118 transactions** after deployment; only 1 was literally next |
| Hold time | median **6 s**; IQR 5 s; p90 17 s; mean 10.62 s |
| Partial exits | **15,612 / 16,163 (96.59%)** |
| Sell structure | median **4 sells/token**; **70,805** sells across bought positions |
| Burns | **2 transactions / 2 tokens** |
| Fully fee-adjusted hit rate | **58.65%** |
| Average winner / loser | **+$117.55 / -$28.30** |
| Fully fee-adjusted P&L | **+$925,056**; mean $57.23; median $8.00 per position |

The fee ledger treats cost_usd as the actual quote principal paid or received. Across the dataset, the wallet paid $4.254941M and received $5.743209M.

The sparse buy_cost_usd field appears on only 22 sell rows. It is treated as a reference or cost-basis field rather than another cash outflow.

Transaction-level gas_usd is subtracted exactly once, for a total of $439,839.55. Since gas_native and gas_usd already include priority and tip components when those values are present, priority_fee and tip_fee are not subtracted again.

Pump trading costs are separate from principal. The target's observed Pump route costs total 1.25% of Pump principal, or $123,373.00. Of this amount, $110,500.77 is represented by the applicable `dex_*` component. Another $12,872.23 is needed to match the total cost visible in the raw data.

The data does not clearly identify what that additional 30 bp transfer represents, so it is not labeled as a creator, referral, or routing fee. Total defensible costs are $563,212.55.

For known routed venues, the observed quote transfer already reflects the wallet's cash flow, so the additional $0.18 of dex_usd is treated as a diagnostic rather than charged again. Blank-venue DEX fields totaling $0.005 and a $4.66 Pump component residual are too ambiguous to count as additional costs.

Refundable rent, unidentified account residuals, remaining inventory, and other unobserved costs are not included. The result should therefore be interpreted as a fee-adjusted cash-flow analysis of the activity table, not a tax ledger or mark-to-market portfolio calculation.

## 3. Legal feature store and preserved control

Both deployment archives are streamed once. Current metadata, dev-buy arguments, ComputeBudget choices, account/instruction structure, and top-level System transfers come only from the signed message; no `meta` field is read. Strict ASOF states require `activity.timestamp < deployment.blockTime`. The store has 5,076,421 unique tokens, 15,927 labels, zero duplicate keys, and zero non-positive recencies.

The activity source timestamps launches only to the second. It has no slot or transaction index; the deployment index adds slots, but 137,880 fully mapped same-wallet/same-second groups still contain same-slot ties. Transaction hash and file order are identifiers/storage order, not time. The former ASOF implementation therefore selected different tied rows in 6,831 candidates. The corrected contract groups every latest launch sharing `(wallet, launch_time)` and uses order-invariant sold/maturity fractions plus the median observed sell latency. All sell visibility checks remain strict-prior to the candidate. No row-level cache or arbitrary hash order is used.

The creator-fee family passes its unchanged pre-June gate. After correcting the tie semantics, developer-sell history adds only +0.000831 April and +0.000665 May PR-AUC; the frozen rule requires at least +0.002 May with no material April regression. It therefore returns `DROP`, and all developer-sell fields are mechanically excluded from the promoted classifier. Organizer monetary strings are accumulated at fixed nine-decimal precision before conversion to model doubles, making source reconstruction independent of parallel floating-point reduction order.

## 4. Promoted target–signer relationship selector

The missing policy signal is live memory of deployment signers the bot previously bought from. For each signer, the model sees only strictly earlier public target buys: cumulative and 1h/6h/1d/7d/30d counts, recency, lifetime and recent conditional rates, relationship age, and launches since the prior target buy. The point-in-time audit reports zero future/equal states, invalid recencies, invalid counts/rates, or duplicate tokens across all 5,076,421 candidates.

Training chronology is fixed and explicit:

- The April gate fits March 12–31 and validates on full April.
- The final frozen model fits **2026-03-12 inclusive through 2026-05-01 exclusive**. April therefore enters the final weights.
- Full May is the final model-promotion and threshold-selection population. May labels do **not** enter the fitted weights.
- The max-F1 threshold is frozen at `0.23211809647507783`.
- June is reporting only. There is no through-May refit and no post-June redesign.

| Chronological result | Prevalence | PR-AUC | Precision | Recall | F1 | Entries | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| April promotion gate | 0.424% | **0.282063** | 32.52% | 52.99% | 0.40303 | 5,557 | 1,807 |
| May promotion/threshold gate | 0.570% | **0.385999** | 41.47% | 64.13% | 0.50367 | 7,852 | 3,256 |
| Canonical corrected June report | 0.492% | **0.2047103771** | **29.32%** | **42.60%** | **0.34736** | **6,094** | **1,787** |

The promoted system beats the preserved final by +0.17465 April AP and +0.23527 May AP. Every daily delta is positive in both months. Excluding target buys from the preceding five seconds changes AP by -0.00320 in April and +0.00409 in May. June PR-AUC is 41.58× prevalence and precision is 59.56× prevalence; absolute precision remains the honest headline under extreme imbalance. The old 0.2082199639 and clean 0.2100330525 June scores are superseded implementation artifacts, not candidate results.

## 5. Reverse-engineered rule and importance

Frozen-model tree gain is descriptive, correlated importance—not a causal effect and not an additive decomposition. The top objective-gain feature, `deployments_since_prior_target_buy`, contributes 58.58% of total gain. It directly represents whether a known signer has launched repeatedly since the bot last chose it. prior_target_buy_rate_30d and prior_target_buy_rate_7d rank fourth and fifth. The retained foundation supplies spacing (`seconds_since_prior_deploy`), signed developer commitment (`dev_buy_sol`), time/regime, recent activity, prior claimed-launch fraction, seven-day cadence, and compute-budget intent.

The inferred rule hierarchy is therefore:

1. **Dominant:** maintain a live, strict-prior relationship state for signers already bought from.
2. **Strong context:** prefer meaningful deployment spacing and non-industrial cadence.
3. **Supporting:** signed dev-buy size, wallet activity/quality history, and compute-budget intent.
4. **Rejected additions:** signed-message identity/archetype recurrence failed its pre-June materiality gate and was never scored on June.

The relationship feature is online, not a static train-only signer whitelist: earlier target transactions during April or May may inform later candidates in the same month, but only after their timestamps. A month-start-frozen sensitivity is much weaker, supporting the live-memory interpretation.

## 6. Classification and economic selection are different tasks

The promoted classifier estimates target-buy probability. Model B separately estimates a creator-fee claim within seven days using only labels mature before each evaluation boundary. After rebuilding its inputs from source, its selective policy makes 1,402 June entries, with 205 target overlaps, 14.62% overlap precision, and 4.89% target recall. That selectivity is a secondary economic strategy choice, not a claim that Model B improves target classification.

We preserve this economic analysis without retuning its feature families, hyperparameters, or operating-point rule. The old classifier likewise remains a secondary control in the exact-execution comparisons below.

## 7. Primary Part 3: source-built marginal-price backtest

| Cohort/policy | Entries | Overlap | Immediate hit / median ROI | +1-slot fill | +1-slot hit / median ROI | +1-slot p99-cap P&L | Drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Target equal-stake diagnostic | 4,195 | — | 64.10% / +15.66% | 99.98% | 32.64% / −13.13% | −369.05 SOL | 338.28 SOL |
| **Corrected Part 2 selector** | **6,094** | **1,787** | **66.74% / +20.27%** | **99.82%** | **34.93% / −10.71%** | **−405.63 SOL** | **332.23 SOL** |

These are marginal observed-price diagnostics with the same 1.9753 SOL stake and six-second exit. They subtract the pre-June target median two-leg **network cost of 0.09101 SOL exactly once**. That inclusive `gas_native` amount already contains priority/tip components. Because this architecture cannot apply proportional Pump fees without unsupported fill assumptions, these results are labeled **network-cost-adjusted and gross of swap fees**, never fully net.

Immediate execution is optimistic. The primary table uses whole-slot delays; it does not claim a mempool view. Increasing delay changes the executable population, so conditional ROI need not decline monotonically. A separate pre-existing +118 landed-position control diagnostic has -4.88% median ROI; its 3,904-token cohort is not the corrected Part 2 selector and remains supporting evidence only.

The target's separate actual June activity ledger has $971,151.32 quote principal paid, $1,290,787.90 received, $105,752.16 inclusive network cost, and $28,274.24 separate Pump cost. Its fully fee-adjusted result is **+$185,610.17**, **57.26%** hit rate, **$5.28** median P&L, and **16.79%** ROI on buys plus defensible costs. Its variable sizing and multi-exit cash flow are not substituted into the equal-stake table.

## 8. Exact Pump mechanics and bounded conclusions

The secondary integer constant-product replay inserts our buy and sell, own price impact, intervening events, proportional swap fees, and the same 0.09101 SOL inclusive network cost. It applies trade principal/curve impact first; then 0.95–1.25% fee assumptions separately on entry and exit; then network cost once. Priority and tips are not added again.

A targeted 169 MB raw batch decodes 82 current-cohort events, confirms fixed-quote and fixed-token buy variants plus 0/30 bp creator fields, and matches every normalized SOL/token amount. At +118 and 0.95% fees:

| Strategy | Supported coverage | Median fully modeled ROI bound | p99-capped P&L bound |
|---|---:|---:|---:|
| Preserved control | 48.54–50.17% | −9.91% to −7.77% | −141.0 to −12.3 SOL |
| Prior quality selector | 43.43–44.89% | −10.41% to −8.50% | −212.4 to −43.4 SOL |
| Equal-count two-stage | 45.28–46.59% | −9.45% to −7.49% | −174.1 to +17.0 SOL |
| **Selective two-stage** | **46.86–47.93%** | **−8.46% to −7.06%** | **−9.0 to +38.2 SOL** |

At 0.95% fees the selective strategy's capped P&L spans a small loss under fixed quote to a gain under fixed token, while its median is negative under both. At 1.25%, the fixed-quote to fixed-token bounds span −17.0 to +29.9 SOL. Its advantage is selectivity and tail behavior, not robust positive median profitability and not proof it beats the bot.

The former exact outcome cache is superseded and remains private: it represented an earlier membership state, with 1,133 replayed tokens outside and 1,141 tokens missing from the then-current union. A clean audit also exposed physical-Parquet-order dependence in the legacy control fit; its candidate order is now frozen as `(block_time, token_address)`, yielding 3,735 control entries. The corrected 1,402-entry selective membership, all exact outcomes, and these aggregates now regenerate from source. No old cache is an input or a publication artifact. Exact replay remains secondary; the source-built marginal backtest is the primary Part 3 claim.

Unsupported nonstandard/Mayhem curves, PumpSwap migrations, exact delayed intent, slippage limits, counterfactual curve completion, rent/refunds, residual accounts, and the target's actual variable-size multi-exit curve path remain unresolved rather than fabricated. No live trading or external execution was performed.
