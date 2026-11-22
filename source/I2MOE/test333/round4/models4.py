"""Local evidence and controlled corrections to the strongest existing E7."""
import torch
from torch import nn
from torch.nn import functional as F


def focal_each(margin: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    y=labels.float()
    p=margin.float().sigmoid()
    pt=torch.where(y.bool(),p,1-p)
    alpha=torch.where(y.bool(),.85,.15)
    return alpha*(1-pt).square()*F.binary_cross_entropy_with_logits(margin.float(),y,reduction='none')


def hard_negative_rank(margin: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive=margin[labels==1]
    negative=margin[labels==0]
    if not len(positive) or not len(negative):return margin.sum()*0
    hard=negative.topk(min(2,len(negative))).values
    return F.softplus(1+hard[None,:]-positive[:,None]).mean()


class LocalMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(200)
        self.gated=nn.Linear(200,96)
        self.values=nn.Linear(200,96)
        self.score=nn.Linear(96,1)
        self.output=nn.Sequential(nn.LayerNorm(400),nn.Linear(400,128),nn.GELU())

    def forward(self,tokens: torch.Tensor) -> torch.Tensor:
        h=self.norm(tokens)
        scores=self.score(torch.tanh(self.gated(h))*torch.sigmoid(self.values(h)))
        pooled=(tokens*scores.softmax(1)).sum(1)
        return self.output(torch.cat((pooled,tokens.amax(1)),-1))


class WaveDetail(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(19,32,7,padding=3),nn.GroupNorm(4,32),nn.GELU())
        self.scales=nn.ModuleList([nn.Conv1d(32,32,k,padding=k//2,groups=32) for k in (7,15,31)])
        self.merge=nn.Sequential(nn.Conv1d(96,64,1),nn.GroupNorm(8,64),nn.GELU(),
                    nn.AvgPool1d(2),nn.Conv1d(64,64,9,padding=8,dilation=2),nn.GroupNorm(8,64),nn.GELU())
        self.output=nn.Sequential(nn.LayerNorm(147),nn.Linear(147,128),nn.GELU())

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        rms=x.square().mean(-1).sqrt().clamp_min(.01)
        z=self.stem(x/rms[:,:,None])
        h=self.merge(torch.cat([layer(z) for layer in self.scales],1))
        return self.output(torch.cat((h.mean(-1),h.amax(-1),rms.log()),-1))


class AnchorCorrection(nn.Module):
    def __init__(self,variant: str,bound: float=.75):
        super().__init__()
        self.variant=variant
        self.bound=bound
        self.encoder=LocalMIL() if variant=='local_mil' else WaveDetail() if variant=='wave_detail' else nn.Sequential(nn.LayerNorm(128),nn.Linear(128,128),nn.GELU())
        dim=264 if variant in ('local_mil','wave_detail') else 136
        self.head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.auxiliary=nn.Linear(128,1) if variant in ('local_mil','wave_detail') else None

    def forward(self,batch: dict) -> dict:
        key='tokens' if self.variant=='local_mil' else 'fm' if self.variant=='wave_detail' else 'features'
        local=self.encoder(batch[key])
        inputs=[local.float()]
        if self.auxiliary is not None:inputs.append(batch['features'].float())
        inputs.extend((batch['base'].float()/4,batch['anchor'].float()[:,None]/4))
        raw=self.head(torch.cat(inputs,-1)).squeeze(-1).float()
        delta=self.bound*torch.tanh(raw/self.bound)
        result=dict(fused=batch['anchor'].float()+delta,delta=delta)
        if self.auxiliary is not None:result['standalone']=self.auxiliary(local).squeeze(-1).float()
        return result


def objective(output: dict,batch: dict,variant: str) -> torch.Tensor:
    margin=output['fused']
    # Express the regularizer through the final margin, so Functional SAM's
    # unperturbed dL/dlogit includes every loss term for its variant.
    loss=focal_each(margin,batch['label']).mean()+.02*(margin-batch['anchor']).square().mean()
    if variant=='hard_rank':loss=loss+.005*hard_negative_rank(margin,batch['label'])
    if 'standalone' in output:loss=loss+.1*focal_each(output['standalone'],batch['label']).mean()
    return loss

