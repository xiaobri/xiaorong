# Global-role ablation results

Audit passed. These are validation-selected results, not independent-test estimates.

Return all six files together: summary.csv, per_seed.csv, paired_differences.csv, diagnostics.csv, report.json, RETURN_RESULTS.md.

All rates are fractions (multiply by 100 for percentages); differences in rates multiplied by 100 are percentage points.
Standard deviations use ddof=1. Missing standard deviations for one seed are empty/null, never zero.

| Arm | Seeds | Mean F1 | Sample SD | Mean precision | Mean recall |
|---|---:|---:|---:|---:|---:|
| A | 5 | 0.687897 | 0.001059 | 0.705206 | 0.671449 |
| B | 5 | 0.688014 | 0.001276 | 0.705099 | 0.671852 |
| C | 5 | 0.696406 | 0.002742 | 0.738896 | 0.658773 |
| D | 5 | 0.697142 | 0.003049 | 0.742286 | 0.657332 |
| E | 5 | 0.697147 | 0.002260 | 0.729079 | 0.667992 |

## Interpretation and required comparisons

Read B-A, C-A, D-A, D-B, D-C, E-D and (D-C)-(B-A) as paired differences for the same seed. The interaction contrast is descriptive because each arm selects its own validation threshold and epoch.

repair_damage rows compare each arm with A at its own selected validation threshold, and separately at fixed 0.5. Repair/damage counts alone do not determine F1 changes.
Top-selected refers to argmax routing weight (first expert in a tie), not an oracle. Expert correctness in these diagnostics uses strict probability > 0.5. Global-alone results can describe a branch excluded from an arm's final prediction.

label_assisted_bin rows partition validation windows using regional correctness at 0.5. They are retrospective descriptions, not training inputs, clinically verified mechanisms, deployment policies, F1 upper bounds, or estimates of irreducible error.

The auditor verified all epoch prediction archives, complete epoch histories, paired initialization/sampling hashes, common arguments and protocol hash, exact validation row order and labels, frozen regional probabilities, normalized finite weights, A/B global masking, and recomputed threshold/F1/confusion counts.

No paths, patient identifiers, or validation row identifiers are included in these exports. No claim of statistical significance or patient-disjointness is made.
