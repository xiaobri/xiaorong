"""Prequential regional evidence from positive/negative pairs seen in the original stream."""
import torch
from torch import nn

from analytic_models import AnalyticRegionMoE, NormalEquations
from allocation_models import method_contract as allocation_contract, method_loss_terms as allocation_loss
from cohort_models import MAX_GROUPS
from i2moe_eeg.losses import task_loss

METHODS = ('pooled_quantiles', 'pooled_nearest', 'within_quantiles', 'within_nearest')


def method_contract(method):
    result = allocation_contract('independent_aux')
    result.update(method=method,additional_streaming_pair_state=True,extra_adamw_steps=0,
        training_population='Only original Train acquisition groups with at least 50 windows of each label contribute to the extra rank estimator. All original windows still train the six-expert parent.',
        pair_definition='Each eligible positive is paired with at most sixteen preceding negatives; store the most recent 2048 eligible negative summaries.',
        domain_restriction='Same acquisition' if method.startswith('within') else 'Pooled eligible acquisitions',
        selection='Closest mean adjusted regional margin' if method.endswith('nearest') else 'Equally spaced quantiles of mean adjusted regional margin',
        score='One regularized logistic Newton step in regional feature differences relative to the parent, centered on observed mixed-group features. Zero until at least four paired positives.',
        full_parent_unchanged=True,negative_memory_read_only_at_evaluation=True,
        evaluation_uses_no_acquisition_or_memory_lookup=True,
        additional_computation='Store prior negative features and accumulate pair normal equations. Potential reuse of prior negatives is additional supervision, with no added EEG rows or AdamW steps.')
    return result


class PairState(nn.Module):
    def __init__(self, width, within, nearest, capacity=2048):
        super().__init__()
        self.within, self.nearest, self.capacity = within, nearest, capacity
        self.calibration = NormalEquations(2)
        self.calibration.coef[0] = 1
        for name,size in [('gram',(width,width)),('rhs',(width,)),('coef',(width,)),
                          ('sum_x',(width,)),('sum_sq',(width,))]:
            self.register_buffer(name,torch.zeros(size,dtype=torch.float64))
        self.register_buffer('negative_x',torch.zeros(capacity,width))
        self.register_buffer('negative_margin',torch.zeros(capacity))
        self.register_buffer('negative_risk',torch.zeros(capacity))
        self.register_buffer('negative_group',torch.full((capacity,),-1,dtype=torch.long))
        self.register_buffer('support',torch.zeros(MAX_GROUPS,dtype=torch.bool))
        for name in ('updates','refresh_step','cursor','pair_count','paired_positives','last_solved'):
            self.register_buffer(name,torch.zeros((),dtype=torch.long))
        self.register_buffer('counts',torch.zeros(2,dtype=torch.long))
        self.register_buffer('eligible_counts',torch.zeros(2,dtype=torch.long))
        self.register_buffer('ready',torch.zeros((),dtype=torch.bool))

    def select(self,group,risk):
        available = self.negative_group == group if self.within else self.negative_group>=0
        ids = torch.where(available)[0]
        k=min(16,len(ids))
        if not k:
            return ids
        if self.nearest:
            return ids[(self.negative_risk[ids]-risk).abs().topk(k,largest=False).indices]
        rank = ((torch.arange(k,device=ids.device)+.5)*len(ids)/k).long()
        return ids[self.negative_risk[ids].argsort()[rank]]

    @torch.no_grad()
    def update(self,features,margin,labels,groups,risk):
        x=features.detach().clamp(-32,32)
        y=labels.double()
        z=torch.stack((margin.double().clamp(-32,32),torch.ones_like(y)),1)
        offset=z@self.calibration.coef
        p=offset.clamp(-12,12).sigmoid()
        h=(p*(1-p)).clamp_min(1e-5)
        self.calibration.add(z,h,h*offset+y-p)
        eligible=(groups>=0)&self.support[groups.clamp_min(0)]
        self.eligible_counts.add_(torch.bincount(labels[eligible],minlength=2))
        self.counts.add_(torch.bincount(labels,minlength=2))
        observed=x[eligible].double()
        self.sum_x.add_(observed.sum(0))
        self.sum_sq.add_(observed.square().sum(0))
        for idx in torch.where(eligible & (labels==1))[0]:
            previous=self.select(groups[idx],risk[idx])
            if not len(previous):
                continue
            dx=(x[idx]-self.negative_x[previous]).double()
            gap=(margin[idx]-self.negative_margin[previous]).double()
            prob=(self.calibration.coef[0]*gap).clamp(-12,12).sigmoid()
            self.gram.add_(dx.T@((prob*(1-prob))[:,None]*dx))
            self.rhs.add_(dx.T@(1-prob))
            self.pair_count.add_(len(previous))
            self.paired_positives.add_(1)
        negatives=torch.where(eligible & (labels==0))[0][-self.capacity:]
        slots=(self.cursor+torch.arange(len(negatives),device=x.device))%self.capacity
        self.negative_x[slots]=x[negatives]
        self.negative_margin[slots]=margin[negatives]
        self.negative_risk[slots]=risk[negatives]
        self.negative_group[slots]=groups[negatives]
        self.cursor.add_(len(negatives)).remainder_(self.capacity)
        self.updates.add_(1)
        if self.updates.item()%8==0:
            self.refresh()

    @torch.no_grad()
    def refresh(self):
        if self.counts[1].item()>=32 and self.counts[0].item()>=128:
            self.calibration.solve()
            self.calibration.coef[0].clamp_(.15,5.)
            self.calibration.coef[1].clamp_(-12,8.)
        if self.paired_positives.item()>=4 and self.last_solved.item()!=self.pair_count.item():
            count=self.eligible_counts.sum()
            variance=(self.sum_sq/count-(self.sum_x/count).square()).clamp_min(1e-4)
            self.coef.copy_(torch.linalg.solve(self.gram+(self.pair_count*variance).diag(),self.rhs))
            self.last_solved.copy_(self.pair_count)
            self.ready.fill_(True)
        self.refresh_step.copy_(self.updates)

    @torch.no_grad()
    def forward(self,features,margin):
        if not self.training and self.refresh_step.item()!=self.updates.item():
            self.refresh()
        if not self.ready.item():
            return torch.zeros_like(margin)
        centered=features.double().clamp(-32,32)-self.sum_x/self.eligible_counts.sum()
        return ((2/self.calibration.coef[0])*(centered@self.coef/2).tanh()).clamp(-6,6).to(margin)


class RiskPairMoE(AnalyticRegionMoE):
    def __init__(self,method,**kwargs):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__('temporal_residual',**kwargs)
        self.pair_method=method
        self.correction.analytic=PairState(5*(self.width+7),method.startswith('within'),method.endswith('nearest'))

    def architecture_summary(self):
        result=super().architecture_summary()
        result.update(method=self.pair_method,method_contract=method_contract(self.pair_method))
        return result


def method_loss_terms(method,output,labels,args,focal):
    parts=allocation_loss('independent_aux',output.allocation_output,labels,args,focal)
    state=output.cohort_state.analytic
    if state.training and torch.is_grad_enabled():
        regional=output.allocation_output.expert_logits[:,:5].detach()
        risk=(regional[...,1]-regional[...,0]).mean(1)
        state.update(output.analytic_features,output.parent_margin,labels,output.training_groups,risk)
    primary=task_loss(output.logits,labels,args.task_loss,focal)
    parts['extra']=parts['extra']+parts['primary']-primary
    parts['primary']=primary
    return parts
