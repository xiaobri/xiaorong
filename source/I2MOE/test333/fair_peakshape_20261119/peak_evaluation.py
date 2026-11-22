"""Report the simultaneous risk_power anchor on exactly the same validation rows."""
from pathlib import Path
import torch
import numpy as np
from protected_evaluation import install as install_core
from i2moe_eeg.metrics import binary_metrics


def install(original):
    install_core(original)
    evaluate=original.evaluate

    def tracked(model,dataset,focal,args,device,epoch):
        values=[]
        def collect(module,inputs,output):
            values.append(output.anchor_output.logits.detach().softmax(-1)[:,1].cpu())
        hook=model.register_forward_hook(collect)
        try:
            metrics=evaluate(model,dataset,focal,args,device,epoch)
        finally:
            hook.remove()
        p=Path(args.output_dir)/f'predictions_epoch{epoch:02d}.npz'
        if p.exists():
            with np.load(p) as data:
                labels=data['labels']
            scores=torch.cat(values).numpy()
            metrics['simultaneous_risk_power']=binary_metrics(labels,scores,original.threshold_grid(args))
        return metrics

    original.evaluate=tracked
