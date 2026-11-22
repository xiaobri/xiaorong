"""Keep original evaluation; additionally record the uncorrected core path."""
import czb_paths as _paths
import bootstrap
from pathlib import Path

import numpy as np

from i2moe_eeg.metrics import binary_metrics


def install(original):
    evaluate = original.evaluate

    def tracked_evaluate(model, dataset, focal, args, device, epoch):
        core_values, corrections = [], []

        def collect(module, inputs, output):
            core_values.append(output.core_output.logits.detach().softmax(-1)[:, 1].cpu().numpy())
            corrections.append(output.correction_margin.detach().cpu().numpy())

        hook = model.register_forward_hook(collect)
        try:
            metrics = evaluate(model, dataset, focal, args, device, epoch)
        finally:
            hook.remove()
        path = Path(args.output_dir)/f'predictions_epoch{epoch:02d}.npz'
        if not path.exists():
            # The original debug evaluation does not save prediction arrays.
            return metrics
        with np.load(path) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        core = np.concatenate(core_values)
        delta = np.concatenate(corrections)
        if len(core) != len(arrays['labels']):
            raise RuntimeError('Core evaluation row count differs')
        arrays.update(core_probabilities=core, correction_margins=delta)
        temporary = path.with_name(path.stem+'.core.tmp.npz')
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
        metrics['core_path'] = binary_metrics(arrays['labels'], core, original.threshold_grid(args))
        metrics['correction_margin'] = dict(mean=float(delta.mean()), std=float(delta.std()),
            min=float(delta.min()), max=float(delta.max()),
            negative_mean=float(delta[arrays['labels'] == 0].mean()),
            positive_mean=float(delta[arrays['labels'] == 1].mean()))
        reference = Path(f'{_paths.RUNS}/I2MOE_test333_fair_cached_20261115/formal/baseline')/f'seed{args.seed}'/path.name
        if reference.exists() and len(core) == 166079:
            with np.load(reference) as baseline:
                np.testing.assert_array_equal(arrays['labels'], baseline['labels'])
                error = np.abs(core.astype(np.float64)-baseline['probabilities'].astype(np.float64))
            metrics['core_vs_separate_baseline'] = dict(maximum_probability_difference=float(error.max()),
                                                       bit_exact=bool((error == 0).all()))
        return metrics

    original.evaluate = tracked_evaluate
