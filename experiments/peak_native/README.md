# 峰形校正头＋原加权排序

从每个脑区的冻结激活计算标准化峰值、带符号的对数均值和对数标准差。五个编码器结合专家与混合分数，输出加到基线分数差上的有界残差。配套训练采用类条件静态采集组权重（上限 64）与批内排序。此处按融合分数选择困难负例。

权重由训练集组频率确定。与 peak 的差值衡量配套训练目标的作用。

## 协和验证结果

| 种子 | 选中轮次 | 锁定阈值 | 阳性窗口 F1（%） | 对同种子基线的增益（百分点） |
|---:|---:|---:|---:|---:|
| 0 | 13 | 0.88 | 71.018 | +1.197 |
| 1 | 14 | 0.89 | 70.267 | +0.759 |
| 2 | 11 | 0.86 | 70.768 | +1.050 |
| 3 | 8 | 0.86 | 70.393 | +1.004 |
| 42 | 10 | 0.90 | 70.873 | +0.688 |

五种子平均 F1 为 **70.664**，种子间样本标准差为 0.320，相对基线的配对平均变化为 **+0.939 个百分点**。五个种子均在本轮从各自的基线初始化训练。

每个检查点的阈值按原 99 点网格选出。表中 F1 根据锁定阈值下 166,079 个原始 EEG 窗口的混淆矩阵计算。[逐种子混淆矩阵](results.csv) · [全部 100 条逐轮缓存验证记录](epochs.csv)

## 文献与实现来源

- 项目用此文作为利用 EEG 中间表示的研究背景。本文献提出的是多基础模型的表示融合。本实验在五个冻结脑区专家上构造自己的读出。 [Jia 等，ICML 2026，EmBrace: A Collective Knowledge Fusion Framework Toward Unified EEG Foundation Models](https://proceedings.mlr.press/v306/jia26d.html)
- 对正负样本分数差使用平滑排序目标借鉴了 RankNet。具体间隔 0.5、损失系数 0.03 和配对规则由本项目设定。 [Burges 等，ICML 2005，Learning to Rank Using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
- 在线困难样本挖掘提供了选择高分负例的思路。本实验在当前批次按分数选至多 16 个负例。 [Shrivastava 等，CVPR 2016，Training Region-Based Object Detectors with Online Hard Example Mining](https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html)
- 监督来自 focal loss，两类的权重和聚焦参数沿用基线。 [Lin 等，ICCV 2017，Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)

五脑区专家与全局冗余专家沿用项目基线。共用骨干和损失的来源见[文献索引](../../REFERENCES.md)。对应实现见 [single.py](../../source/I2MOE/test333/fair_single_components/single.py) · [peak_models.py](../../source/I2MOE/test333/fair_peakshape_20261119/peak_models.py) · [risk_models.py](../../source/I2MOE/test333/fair_riskrefine_20261115/risk_models.py)。

## 运行

从仓库根目录核验本组公开结果只需要 Python 标准库。准备好私有数据和参考训练记录后，可用同一入口训练指定种子。环境变量与缓存准备见[复现说明](../../REPRODUCING.md)。

```bash
python3 experiments/peak_native/run.py --verify
CUDA_VISIBLE_DEVICES=0 python experiments/peak_native/run.py --seed 3
```
