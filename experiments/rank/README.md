# 批内困难负例排序

在原融合分数上增加 0.03 倍的正负配对排序损失。每个阳性使用同批最高分的至多 16 个负例，单样本权重固定为 1。

负例来自当前批次，历史配对队列是另一项实验。

## 协和验证结果

| 种子 | 选中轮次 | 锁定阈值 | 阳性窗口 F1（%） | 对同种子基线的增益（百分点） |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0.98 | 69.879 | +0.058 |
| 1 | 6 | 0.96 | 69.356 | -0.152 |
| 2 | 1 | 0.99 | 69.835 | +0.116 |
| 3 | 2 | 0.98 | 69.769 | +0.379 |
| 42 | 1 | 0.96 | 69.853 | -0.331 |

五种子平均 F1 为 **69.738**，种子间样本标准差为 0.217，相对基线的配对平均变化为 **+0.014 个百分点**。五个种子均在本轮从各自的基线初始化训练。

每个检查点的阈值按原 99 点网格选出。表中 F1 根据锁定阈值下 166,079 个原始 EEG 窗口的混淆矩阵计算。[逐种子混淆矩阵](results.csv) · [全部 100 条逐轮缓存验证记录](epochs.csv)

## 文献与实现来源

- 对正负样本分数差使用平滑排序目标借鉴了 RankNet。具体间隔 0.5、损失系数 0.03 和配对规则由本项目设定。 [Burges 等，ICML 2005，Learning to Rank Using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
- 在线困难样本挖掘提供了选择高分负例的思路。本实验在当前批次按分数选至多 16 个负例。 [Shrivastava 等，CVPR 2016，Training Region-Based Object Detectors with Online Hard Example Mining](https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html)

五脑区专家与全局冗余专家沿用项目基线。共用骨干和损失的来源见[文献索引](../../REFERENCES.md)。对应实现见 [single.py](../../source/I2MOE/test333/fair_single_components/single.py) · [risk_models.py](../../source/I2MOE/test333/fair_riskrefine_20261115/risk_models.py)。

## 运行

从仓库根目录核验本组公开结果只需要 Python 标准库。准备好私有数据和参考训练记录后，可用同一入口训练指定种子。环境变量与缓存准备见[复现说明](../../REPRODUCING.md)。

```bash
python3 experiments/rank/run.py --verify
CUDA_VISIBLE_DEVICES=0 python experiments/rank/run.py --seed 3
```
