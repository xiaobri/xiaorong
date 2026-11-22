# 历史正负样本配对

在同一采集组内，将每个合格阳性与历史队列中按区域风险分位数取出的至多 16 个负例配对。队列容量为 2,048。用冻结特征及其时间描述量的差值更新校正器，推理时只读取已拟合状态。配对时沿用样本采集当时保存的基线分数。

同采集组限制、风险分位选样、具体容量和正则化更新是本项目的组合设计。

## 协和验证结果

| 种子 | 选中轮次 | 锁定阈值 | 阳性窗口 F1（%） | 对同种子基线的增益（百分点） |
|---:|---:|---:|---:|---:|
| 0 | 10 | 0.81 | 72.334 | +2.514 |
| 1 | 8 | 0.79 | 72.138 | +2.629 |
| 2 | 11 | 0.85 | 71.669 | +1.950 |
| 3 | 8 | 0.79 | 71.867 | +2.478 |
| 42 | 15 | 0.74 | 71.670 | +1.485 |

五种子平均 F1 为 **71.936**，种子间样本标准差为 0.294，相对基线的配对平均变化为 **+2.211 个百分点**。五个种子均在本轮从各自的基线初始化训练。

每个检查点的阈值按原 99 点网格选出。表中 F1 根据锁定阈值下 166,079 个原始 EEG 窗口的混淆矩阵计算。[逐种子混淆矩阵](results.csv) · [全部 100 条逐轮缓存验证记录](epochs.csv)

## 文献与实现来源

- 跨批次保留已经见过的样本表示，为历史负例记忆提供了先例。 [Wang 等，CVPR 2020，Cross-Batch Memory for Embedding Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html)
- 利用冻结特征在线累积统计量、更新读出器，是本项目解析校正的参考。此处采用配对差值的正则化逻辑更新。 [McDonnell 等，NeurIPS 2023，RanPAC: Random Projections and Pre-trained Models for Continual Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2793dc35e14003dd367684d93d236847-Abstract-Conference.html)

五脑区专家与全局冗余专家沿用项目基线。共用骨干和损失的来源见[文献索引](../../REFERENCES.md)。对应实现见 [single.py](../../source/I2MOE/test333/fair_single_components/single.py) · [pair_models.py](../../source/I2MOE/test333/fair_risk_pairs_20261120/pair_models.py) · [analytic_models.py](../../source/I2MOE/test333/fair_analytic_regions_20261120/analytic_models.py)。

## 运行

从仓库根目录核验本组公开结果只需要 Python 标准库。准备好私有数据和参考训练记录后，可用同一入口训练指定种子。环境变量与缓存准备见[复现说明](../../REPRODUCING.md)。

```bash
python3 experiments/pairs/run.py --verify
CUDA_VISIBLE_DEVICES=0 python experiments/pairs/run.py --seed 3
```
