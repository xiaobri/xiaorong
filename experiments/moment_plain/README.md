# 区域统计校正头

读出每个脑区末端激活的均值、最大值和标准差，连同专家分数、混合分数与路由特征输入校正头。核心六专家保留原目标，校正分支读取经梯度分离的输入。独立头只用普通 focal 监督。

统计量组合、分数校正及梯度分离由本项目设计。

## 协和验证结果

| 种子 | 选中轮次 | 锁定阈值 | 阳性窗口 F1（%） | 对同种子基线的增益（百分点） |
|---:|---:|---:|---:|---:|
| 0 | 9 | 0.82 | 70.100 | +0.279 |
| 1 | 12 | 0.81 | 69.363 | -0.146 |
| 2 | 1 | 0.91 | 69.496 | -0.223 |
| 3 | 7 | 0.83 | 69.616 | +0.227 |
| 42 | 10 | 0.83 | 69.410 | -0.774 |

五种子平均 F1 为 **69.597**，种子间样本标准差为 0.297，相对基线的配对平均变化为 **-0.127 个百分点**。seed 3 和 42 复用此前已完成原始 EEG 复核的 plain 单阶段实验。seed 0、1、2 在本轮新训。

每个检查点的阈值按原 99 点网格选出。表中 F1 根据锁定阈值下 166,079 个原始 EEG 窗口的混淆矩阵计算。[逐种子混淆矩阵](results.csv) · [全部 100 条逐轮缓存验证记录](epochs.csv)

## 文献与实现来源

- 监督来自 focal loss，两类的权重和聚焦参数沿用基线。 [Lin 等，ICCV 2017，Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)
- 项目用此文作为利用 EEG 中间表示的研究背景。本文献提出的是多基础模型的表示融合。本实验在五个冻结脑区专家上构造自己的读出。 [Jia 等，ICML 2026，EmBrace: A Collective Knowledge Fusion Framework Toward Unified EEG Foundation Models](https://proceedings.mlr.press/v306/jia26d.html)

五脑区专家与全局冗余专家沿用项目基线。共用骨干和损失的来源见[文献索引](../../REFERENCES.md)。对应实现见 [single.py](../../source/I2MOE/test333/fair_single_components/single.py) · [protected_models.py](../../source/I2MOE/test333/fair_protected_20261115/protected_models.py) · [complement_models.py](../../source/I2MOE/test333/fair_complement_20261118/complement_models.py)。

## 运行

从仓库根目录核验本组公开结果只需要 Python 标准库。准备好私有数据和参考训练记录后，可用同一入口训练指定种子。环境变量与缓存准备见[复现说明](../../REPRODUCING.md)。

```bash
python3 experiments/moment_plain/run.py --verify
CUDA_VISIBLE_DEVICES=0 python experiments/moment_plain/run.py --seed 3
```
