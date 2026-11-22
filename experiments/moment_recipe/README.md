# 区域统计校正头＋加权排序

读出每个脑区末端激活的均值、最大值和标准差，连同专家分数、混合分数与路由特征输入校正头。核心六专家保留原目标，校正分支读取经梯度分离的输入。对校正后的分数启用自适应组加权与批内困难负例排序。

这行测量校正头及其配套训练的整体效果。与 moment_plain 相减可估计更换校正头训练目标的影响。

## 协和验证结果

| 种子 | 选中轮次 | 锁定阈值 | 阳性窗口 F1（%） | 对同种子基线的增益（百分点） |
|---:|---:|---:|---:|---:|
| 0 | 9 | 0.88 | 71.028 | +1.207 |
| 1 | 7 | 0.94 | 71.056 | +1.548 |
| 2 | 9 | 0.88 | 71.508 | +1.790 |
| 3 | 8 | 0.82 | 71.484 | +2.094 |
| 42 | 10 | 0.85 | 71.812 | +1.628 |

五种子平均 F1 为 **71.378**，种子间样本标准差为 0.333，相对基线的配对平均变化为 **+1.653 个百分点**。seed 3 和 42 复用此前已完成原始 EEG 复核的 risk_power 单阶段实验。seed 0、1、2 在本轮新训。

每个检查点的阈值按原 99 点网格选出。表中 F1 根据锁定阈值下 166,079 个原始 EEG 窗口的混淆矩阵计算。[逐种子混淆矩阵](results.csv) · [全部 100 条逐轮缓存验证记录](epochs.csv)

## 文献与实现来源

- 按采集组强调高风险样本借鉴了 Group DRO 的问题设定。实现采用项目定义的类条件逆频率、64 的权重上限以及组风险移动平均。 [Sagawa 等，ICLR 2020，Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731)
- 对正负样本分数差使用平滑排序目标借鉴了 RankNet。具体间隔 0.5、损失系数 0.03 和配对规则由本项目设定。 [Burges 等，ICML 2005，Learning to Rank Using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
- 在线困难样本挖掘提供了选择高分负例的思路。本实验在当前批次按分数选至多 16 个负例。 [Shrivastava 等，CVPR 2016，Training Region-Based Object Detectors with Online Hard Example Mining](https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html)

五脑区专家与全局冗余专家沿用项目基线。共用骨干和损失的来源见[文献索引](../../REFERENCES.md)。对应实现见 [single.py](../../source/I2MOE/test333/fair_single_components/single.py) · [complement_models.py](../../source/I2MOE/test333/fair_complement_20261118/complement_models.py) · [cohort_models.py](../../source/I2MOE/test333/fair_cohorts_20261115/cohort_models.py) · [risk_models.py](../../source/I2MOE/test333/fair_riskrefine_20261115/risk_models.py)。

## 运行

从仓库根目录核验本组公开结果只需要 Python 标准库。准备好私有数据和参考训练记录后，可用同一入口训练指定种子。环境变量与缓存准备见[复现说明](../../REPRODUCING.md)。

```bash
python3 experiments/moment_recipe/run.py --verify
CUDA_VISIBLE_DEVICES=0 python experiments/moment_recipe/run.py --seed 3
```
