# Single additions to the six-expert EEG baseline

The outcome is positive-class window F1 on the original Xiehe validation split. Each row uses the same five seeds (0, 1, 2, 3, 42). Within seed, the baseline and every fit share the core initialization, five frozen regional checkpoints, manifests, order of all 400,000 training draws, batch size 128, and optimizer. Each fit selects its own checkpoint from 20 epochs and its threshold from the original 99-point grid. Raw EEG is scored again at that locked operating point. Report percentage-point differences within seed before averaging.

| Arm | Change applied directly to the baseline |
| --- | --- |
| moment_plain | Add the original detached residual head reading regional mean/max/SD, expert scores and router inputs. Supervise the corrected score with ordinary focal loss. |
| moment_recipe | Add the same head with its published cap-64, class-conditional acquisition weighting, moving group-risk estimate and 0.03 top-16 minibatch ranking loss. |
| weight | Replace only the original fused focal term with its acquisition-weighted version. The global-expert and interaction objectives retain their coefficients. |
| rank | Add 0.03 top-16 hard-negative ranking to the original fused score with unit weights. |
| peak | Add the published standardized peak/mean/SD residual to the baseline margin. Train its score with ordinary focal loss. |
| regional_time | Reweight eight temporal segments in each of five frozen regional experts. Train through the final score with ordinary focal loss. |
| time | Enable both regional temporal reweighting and the corresponding global-expert residual. Use ordinary focal loss. |
| pairs | Add only the within-acquisition historical pair correction. Use original baseline features, unadjusted regional margins and collection-time baseline scores. |
| pairs_refresh | Add the same pair correction and re-evaluate case and historical control scores with the current baseline in evaluation mode. |
| aux_only | Add 0.1 times the original regional auxiliary focal loss with its static group weights to the five frozen classifier outputs. |
| peak_native | Attach the peak head directly to the baseline and use the head's original static weighting and 0.03 top-16 ranking. |
| time_native | Attach temporal reweighting with the global residual directly to the baseline. Use its original static weighting and 0.03 risk-matched ranking. |
| time_aux | Add the regional auxiliary loss to time_native. This estimates its effect after the regional pooling parameters are trainable. |

The contrast time minus regional_time estimates the global residual conditional on temporal changes. With fixed native pooling its two inputs (regional representation changes and time-weight changes) are zero, and both linear maps are bias-free. The contrast pairs_refresh minus pairs estimates the incremental effect of refreshing scores with a populated pair memory. The regional auxiliary objective on the original frozen regions has zero gradient with respect to every trainable parameter. The CPU regression tests check both this identity and the zero-input global residual.

The seed-3 and seed-42 rows for moment_plain and moment_recipe use the published raw single-stage results for plain and risk_power, respectively. All other intervention rows are fresh fits. Reference values for the five original baselines and complete models come from the locked five-seed replay. Report the complete model as a reference and report dependent contrasts separately. These differences depend on the attachment point and cannot be summed to reconstruct the complete model's gain.

To reproduce, use the project's original environment and existing original and additional-seed caches. Run queue_fits.py, then replay.py with --study, --seed and --shard for each of the eight shards, then report.py with --study pointing to the released CZB-MOE checkout. Aggregate files are single_component_seeds.csv, single_component_summary.csv and dependent_contrasts.csv beneath the experiment output directory.
