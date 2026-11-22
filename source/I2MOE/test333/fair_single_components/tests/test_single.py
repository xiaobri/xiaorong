from types import SimpleNamespace
import unittest

import torch

from single import AllocationLayer, SingleComponent, base_weights, loss
from i2moe_eeg.f1_variants import loss_terms
from i2moe_eeg.losses import MulticlassFocalLoss


class InterventionTests(unittest.TestCase):
    def test_aux_on_frozen_experts_has_no_trainable_gradient(self):
        torch.manual_seed(7)
        n=8
        regional=torch.randn(n,5,2)
        learnable=torch.randn(n,1,2,requires_grad=True)
        router=torch.randn(n,6,requires_grad=True)
        experts=torch.cat((regional,learnable),1)
        logits=(experts*router.softmax(-1)[:,:,None]).sum(1)
        core=SimpleNamespace(logits=logits,expert_logits=experts,interaction_view_features=None)
        labels=torch.arange(n)%2
        out=SimpleNamespace(core_output=core,logits=logits,expert_logits=experts,
                            training_groups=torch.zeros(n,dtype=torch.long),
                            cohort_state=SimpleNamespace(group_base_weights=torch.ones(2,1)))
        args=SimpleNamespace(task_loss='focal',variant='baseline',auxiliary_task_loss_weight=.05,
                             interaction_loss_weight=.005)
        focal=MulticlassFocalLoss(.85,2.)
        base=loss_terms(core,labels,args,focal)['total']
        extra=loss('aux_only',out,labels,args,focal)['total']
        self.assertGreater(extra.item(),base.item())
        p=(router,learnable)
        original=torch.autograd.grad(base,p,retain_graph=True)
        actual=torch.autograd.grad(extra,p)
        for a,b in zip(actual,original):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_global_adjustment_without_temporal_shift_is_inert(self):
        layer=AllocationLayer(128)
        for p in layer.redundancy.parameters():
            torch.nn.init.normal_(p)
        value=layer.redundancy(torch.zeros(4,5*(128+8)))
        torch.testing.assert_close(value,torch.zeros_like(value),rtol=0,atol=0)
        value.sum().backward()
        self.assertTrue(all(torch.count_nonzero(p.grad)==0 for p in layer.redundancy.parameters()))


if __name__=='__main__':
    unittest.main()
