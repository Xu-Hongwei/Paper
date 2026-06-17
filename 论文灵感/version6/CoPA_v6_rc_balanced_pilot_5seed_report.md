# CoPA-v6 RC-BalancedAdd 5-Seed Pilot Report

> Purpose: decide whether the hard relation-conditioned BalancedAdd pilot should be expanded to 15 seeds.

## Run Set

Use these five effective runs:

```text
seed 1: code/disagreement_phenomenon/outputs/mosei/multi_seed_20260617_103454/runs/mosei/20260617_103500
seed 2: code/disagreement_phenomenon/outputs/mosei/multi_seed_20260617_103454/runs/mosei/20260617_104346
seed 3: code/disagreement_phenomenon/outputs/mosei/multi_seed_20260617_103454/runs/mosei/20260617_105310
seed 4: code/disagreement_phenomenon/outputs/mosei/20260617_110619
seed 5: code/disagreement_phenomenon/outputs/mosei/20260617_111558
```

Note: the original 5-seed multi-seed command timed out after writing seed 1-3. Seed 3 was also rerun separately at `20260617_105806`, but this duplicate is not included in the pilot summary.

## Main Result

All values are Macro-F1 deltas relative to Concat unless noted.

| Mode | Group | RC Delta | 95% CI | Positive Rate | Fixed BalancedDirectAdd Delta | RC - Fixed Balanced |
|---|---:|---:|---:|---:|---:|---:|
| rd_only | RD | -0.0046 | [-0.0173, 0.0082] | 40% | +0.0091 | -0.0137 |
| hard | RD | +0.0074 | [-0.0081, 0.0230] | 80% | +0.0091 | -0.0017 |
| rd_only | ND | -0.0011 | [-0.0112, 0.0091] | 40% | -0.0021 | +0.0010 |
| hard | ND | +0.0105 | [0.0036, 0.0175] | 100% | -0.0021 | +0.0126 |
| rd_only | UA | -0.0017 | [-0.0230, 0.0195] | 40% | +0.0167 | -0.0184 |
| hard | UA | -0.0103 | [-0.0418, 0.0213] | 60% | +0.0167 | -0.0270 |

Overall:

| Mode | Overall RC Delta | 95% CI | Positive Rate | Overall RC - Fixed Balanced |
|---|---:|---:|---:|---:|
| rd_only | +0.0003 | [-0.0084, 0.0091] | 40% | -0.0044 |
| hard | +0.0032 | [-0.0066, 0.0131] | 80% | -0.0015 |

## Interpretation

`RD-Only BalancedAdd` should not be expanded. It is weaker than fixed BalancedDirectAdd on RD in all 5 seeds.

`RC-BalancedAdd-Hard` is a mixed result. It has positive mean RD delta and 4/5 positive RD seeds, but it does not beat fixed BalancedDirectAdd on average. The RD improvement is also not error-controlled in this 5-seed pilot.

The hard schedule has a visible side effect: it improves ND relative to fixed BalancedDirectAdd, but it underperforms fixed BalancedDirectAdd on UA. This suggests that the current hard alpha map is not a clean relation-state scheduling rule.

## Decision

Do not expand `RC-BalancedAdd-Hard` to 15 seeds yet.

The current paper should keep BalancedDirectAdd as the minimal positive clue and write this pilot, if used at all, as a negative/boundary result:

```text
Hard relation-conditioned alpha scheduling is not sufficient to outperform fixed balanced utilization.
This supports the need for a learned or soft relation gate rather than a manually fixed hard rule.
```

## Next Action

For the current motivation-first paper draft:

```text
Use BalancedDirectAdd as the bridge evidence.
Do not claim RC-BalancedAdd success.
Do not run more hard-gate seeds now.
Only consider soft gate after the motivation section is written and the hard-gate negative result is clearly framed.
```

