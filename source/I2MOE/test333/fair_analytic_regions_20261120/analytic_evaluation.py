from pathlib import Path

import numpy as np
import torch

from field_evaluation import install as install_reference
from i2moe_eeg.metrics import binary_metrics


def install(original):
    install_reference(original)
    evaluate=original.evaluate

    def tracked(model,dataset,focal,args,device,epoch):
        margins=[]
        def capture(module,inputs,out):
            margins.append(out.parent_margin.detach().cpu())
        handle=model.register_forward_hook(capture)
        try:
            result=evaluate(model,dataset,focal,args,device,epoch)
        finally:
            handle.remove()
        file=Path(args.output_dir)/f'predictions_epoch{epoch:02d}.npz'
        if file.exists():
            with np.load(file) as saved:
                labels=saved['labels']
            result['simultaneous_allocation_anchor']=binary_metrics(labels,torch.cat(margins).sigmoid().numpy(),
                                                                  original.threshold_grid(args))
        state=model.correction.analytic
        result['analytic_learning']=dict(batches=state.updates.item(),counts=state.counts.cpu().tolist(),
            last_refresh=state.refresh_step.item(),active=bool(state.ready.item()),
            calibration=state.calibration.coef.cpu().tolist())
        return result

    original.evaluate=tracked
