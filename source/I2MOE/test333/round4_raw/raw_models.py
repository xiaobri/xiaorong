"""One hybrid E7: original CBraMod anchor plus a raw-waveform detail branch."""
import torch
from torch import nn
from torch.nn import functional as F
from models4 import focal_each,hard_negative_rank


def supervised_contrastive(features,labels,temperature=.2):
    z=F.normalize(features.float(),dim=-1)
    similarity=z@z.T/temperature
    off_diagonal=~torch.eye(len(z),dtype=torch.bool,device=z.device)
    logits=similarity.masked_fill(~off_diagonal,-1e9)
    log_prob=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    positive=(labels[:,None]==labels[None,:])&off_diagonal
    count=positive.sum(1)
    per_anchor=-(log_prob*positive).sum(1)/count.clamp_min(1)
    terms=[]
    for label in (0,1):
        valid=(labels==label)&(count>0)
        if valid.any():terms.append(per_anchor[valid].mean())
    return torch.stack(terms).mean() if terms else features.sum()*0


class RawEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        frequency=torch.fft.rfftfreq(2000,d=1/500)
        mask=(frequency>=1)&(frequency<=200)
        for line in (50,100,150,200):mask=mask&((frequency-line).abs()>1)
        self.register_buffer('frequency_mask',mask.float()[None,None,:])
        self.banks=nn.ModuleList([nn.Conv1d(19,38,k,stride=4,padding=k//2,groups=19) for k in (7,15,31)])
        self.spatial=nn.Sequential(nn.GroupNorm(19,114),nn.GELU(),nn.Conv1d(114,96,1),
                                  nn.GroupNorm(12,96),nn.GELU(),nn.AvgPool1d(2))
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(96,96,9,padding=4*d,dilation=d,groups=96),
                      nn.GroupNorm(12,96),nn.GELU(),nn.Conv1d(96,96,1)) for d in (1,4)])
        self.attention=nn.Sequential(nn.Conv1d(96,32,1),nn.Tanh(),nn.Conv1d(32,1,1))
        self.output=nn.Sequential(nn.LayerNorm(211),nn.Linear(211,128),nn.GELU())

    def filtered(self,x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            return torch.fft.irfft(torch.fft.rfft(x.float(),dim=-1)*self.frequency_mask,n=2000,dim=-1)

    def forward(self,raw):
        x=self.filtered(raw)
        rms=x.square().mean(-1).sqrt().clamp_min(.01)
        normalized=x/rms[:,:,None]
        parts=[bank(normalized).reshape(len(x),19,2,500) for bank in self.banks]
        h=self.spatial(torch.cat(parts,dim=2).reshape(len(x),114,500))
        for block in self.blocks:h=F.gelu(h+block(h))
        weighted=(h*self.attention(h).softmax(-1)).sum(-1)
        return self.output(torch.cat((weighted,h.amax(-1),rms.log()),-1))


class RawCorrection(nn.Module):
    def __init__(self,variant,bound=.75):
        super().__init__()
        self.variant=variant;self.bound=bound
        self.encoder=RawEncoder()
        self.head=nn.Sequential(nn.LayerNorm(264),nn.Linear(264,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
        self.auxiliary=nn.Linear(128,1)
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,batch):
        embedding=self.encoder(batch['raw'])
        features=torch.cat((embedding.float(),batch['features'].float(),batch['base'].float()/4,
                            batch['anchor'].float()[:,None]/4),-1)
        raw=self.head(features).squeeze(-1).float()
        delta=self.bound*torch.tanh(raw/self.bound)
        return dict(fused=batch['anchor'].float()+delta,delta=delta,embedding=embedding,
                    standalone=self.auxiliary(embedding).squeeze(-1).float())


def objective(output,batch,variant):
    margin=output['fused']
    loss=focal_each(margin,batch['label']).mean()+.02*(margin-batch['anchor']).square().mean()
    loss=loss+.1*focal_each(output['standalone'],batch['label']).mean()
    if variant=='raw_rank':loss=loss+.005*hard_negative_rank(margin,batch['label'])
    elif variant=='raw_contrast':loss=loss+.003*supervised_contrastive(output['embedding'],batch['label'])
    else:raise ValueError(variant)
    return loss

