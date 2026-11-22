"""Exercise both refresh paths on actual previously observed baseline training packets."""
import czb_paths as _paths
import argparse
import json
import numpy as np
import torch

import bootstrap
import train_eeg19_i2moe as trainer
from cohort_execution import make_metadata
from i2moe_eeg.utils import atomic_write_json
from refresh_execution import history_batch, install
from refresh_models import RefreshedPairMoE


def main() -> None:
    torch.set_num_threads(2)
    path = f'{_paths.STORAGE}/I2MOE6_F1/screen_20261112/baseline/seed3/args.json'
    args = argparse.Namespace(**json.loads(open(path).read()))
    device = trainer.initialize_device(args)
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(device).total_memory)
    frame, _ = trainer.load_manifest(args.csv_path)
    meta, groups = make_metadata(frame)
    install(trainer, 3, 'fresh_both')
    data = trainer.EEG19NpyDataset(frame,'Train',args.stored_prefix,args.data_root,
        target_samples=args.target_samples,eeg_scale_divisor=args.eeg_scale_divisor)
    data.set_epoch(1)
    trainer.SixExpertF1 = lambda **kw: RefreshedPairMoE('fresh_both',**kw)
    model = trainer.build_model(args).to(device).train()
    state = model.correction.analytic
    supported = (np.array(meta['counts'])>=50).all(0)
    state.support.copy_(torch.as_tensor(supported))
    order, _ = trainer.sampled_indices(data.labels,20000,30,'match_cam',128,3,1)
    order = np.array(list(torch.utils.data.SubsetRandomSampler(order.tolist(),
                          generator=torch.Generator().manual_seed(10003))))
    eligible = order[(data.labels[order]==1) & supported[groups[order]]]
    pos = next(i for i in eligible if np.sum((data.labels[order]==0) & (groups[order]==groups[i]))>=128)
    neg = order[(data.labels[order]==0) & (groups[order]==groups[pos])][:128]
    def forward(rows):
        batch, labels = torch.utils.data.default_collate([data[int(i)] for i in rows])
        output = model(batch.to(device),compute_interactions=False)
        return batch.to(device), labels.to(device), output
    packet, labels, out = forward(neg)
    out.refresh_history(out,labels)
    risk = out.allocation_output.expert_logits[:,:5].diff(dim=-1).mean(1).flatten()
    state.update(out.analytic_features,out.parent_margin,labels,out.training_groups,risk)
    np.testing.assert_array_equal(state.row_ids[:128].cpu().numpy(),neg)
    data.set_epoch(2)
    reread = history_batch(neg[:16], [1]*16)
    torch.testing.assert_close(reread,packet[:16].cpu(),rtol=0,atol=0)
    assert data.original_epoch == 2
    data.set_epoch(1)
    packet, labels, out = forward([pos])
    expected_case = model.current_parent(packet)
    selected = state.select(out.training_groups[0],state.negative_risk[0])
    expected_controls = model.current_parent(history_batch(state.row_ids[selected].cpu().tolist(),[1]*len(selected)).to(device))
    state.negative_margin[selected] += 10
    flags = [m.training for m in model.modules()]
    out.refresh_history(out,labels)
    assert flags == [m.training for m in model.modules()]
    torch.testing.assert_close(state.negative_margin[selected],expected_controls)
    torch.testing.assert_close(out.parent_margin,expected_case)
    assert (state.refreshed_cases.item(),state.refreshed_controls.item()) == (1,16)
    atomic_write_json(bootstrap.OUTPUT/'actual_history_preflight.json',dict(verified=True,
        existing_negative_windows=128, exact_original_epoch_reread=16, controls_refreshed=16,
        cases_refreshed=1, original_training_modes_restored=True, row_and_epoch_retained=True))


if __name__=='__main__':
    main()
