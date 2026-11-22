from pathlib import Path

import numpy as np
import torch

from peak_evaluation import install as install_anchors
from i2moe_eeg.metrics import binary_metrics


def install(original):
    install_anchors(original)
    evaluate = original.evaluate

    def tracked(model,dataset,focal,args,device,epoch):
        peaks = []
        def capture(module,inputs,out):
            peaks.append(out.peak_output.logits.detach().softmax(-1)[:,1].cpu())
        handle = model.register_forward_hook(capture)
        try:
            result = evaluate(model,dataset,focal,args,device,epoch)
        finally:
            handle.remove()
        path = Path(args.output_dir)/f'predictions_epoch{epoch:02d}.npz'
        if path.exists():
            with np.load(path) as data:
                y = data['labels']
            scores = torch.cat(peaks).numpy()
            result['simultaneous_peak_anchor'] = binary_metrics(y,scores,original.threshold_grid(args))
        return result

    original.evaluate = tracked
