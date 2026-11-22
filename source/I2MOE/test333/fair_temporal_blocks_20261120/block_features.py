"""Pre-LayerNorm regional feature segments for exact linear re-pooling."""
import torch

SEGMENTS=8


def base_weights(length,reference):
    if length<SEGMENTS:
        raise ValueError('Feature trace too short for segmented pooling')
    sizes=[length//SEGMENTS+int(k<length%SEGMENTS) for k in range(SEGMENTS)]
    return reference.new_tensor(sizes)/length


def projected_blocks(features,projection):
    trace=features.mean(-2)
    blocks=torch.stack([t.mean(-1) for t in trace.tensor_split(SEGMENTS,dim=-1)],1)
    return projection[0](blocks)


def register_blocks(experts):
    captured,handles={},[]
    for index,expert in enumerate(experts.values()):
        source=getattr(expert,'module',expert)
        def record(module,args,out,index=index,source=source):
            captured[index]=(projected_blocks(out,source.encoder.projection).detach(),out.shape[-1])
        handles.append(source.encoder.features.register_forward_hook(record))
    return captured,handles
