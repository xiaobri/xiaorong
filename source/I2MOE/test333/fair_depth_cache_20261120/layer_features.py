"""Capture channel statistics at the same three original EfficientNet depths."""
import torch

STAGES = (2,4,6)
WIDTHS = (48,128,256)


def summarize(values):
    values = values.flatten(2)
    return torch.stack((values.mean(-1),values.amax(-1),values.std(-1,unbiased=False)),1)


def register_statistics(experts, taps=STAGES):
    captured,handles = {},[]
    for region,expert in enumerate(experts.values()):
        source = getattr(expert,'module',expert)
        for idx,stage in enumerate(taps):
            def record(module,args,output,region=region,idx=idx):
                captured[region,idx] = summarize(output).detach()
            handles.append(source.encoder.features[stage].register_forward_hook(record))
    return captured,handles


def pack_statistics(captured):
    return torch.stack([torch.cat([captured[region,i].flatten(1) for i in range(3)],1)
                        for region in range(5)],1)
