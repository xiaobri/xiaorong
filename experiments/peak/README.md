# 峰形校正头

从每个脑区的冻结激活计算标准化峰值、带符号的对数均值和对数标准差。五个编码器结合专家与混合分数，输出加到基线分数差上的有界残差。此处使用普通 focal 监督。

峰形描述量和残差头由本项目设计。这一分支使用当前窗口的实测统计量。

## 协和验证结果

| 种子 | 选中轮次 | 锁定阈值 | 阳性窗口 F1（%） | 对同种子基线的增益（百分点） |
|---:|---:|---:|---:|---:|
| 0 | 9 | 0.82 | 69.857 | +0.036 |
| 1 | 14 | 0.80 | 69.412 | -0.096 |
| 2 | 13 | 0.82 | 69.723 | +0.004 |
| 3 | 2 | 0.82 | 69.373 | -0.017 |
| 42 | 7 | 0.82 | 70.146 | -0.039 |

五种子平均 F1 为 **69.702**，种子间样本标准差为 0.322，相对基线的配对平均变化为 **-0.022 个百分点**。五个种子均在本轮从各自的基线初始化训练。

每个检查点的阈值按原 99 点网格选出。表中 F1 根据锁定阈值下 166,079 个原始 EEG 窗口的混淆矩阵计算。[逐种子混淆矩阵](results.csv) · [全部 100 条逐轮缓存验证记录](epochs.csv)

## 文献与实现来源

- 项目用此文作为利用 EEG 中间表示的研究背景。本文献提出的是多基础模型的表示融合。本实验在五个冻结脑区专家上构造自己的读出。 [Jia 等，ICML 2026，EmBrace: A Collective Knowledge Fusion Framework Toward Unified EEG Foundation Models](https://proceedings.mlr.press/v306/jia26d.html)
- 监督来自 focal loss，两类的权重和聚焦参数沿用基线。 [Lin 等，ICCV 2017，Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)

五脑区专家与全局冗余专家沿用项目基线。共用骨干和损失的来源见[文献索引](../../REFERENCES.md)。对应实现见 [single.py](../../source/I2MOE/test333/fair_single_components/single.py) · [peak_models.py](../../source/I2MOE/test333/fair_peakshape_20261119/peak_models.py)。

## 运行

从仓库根目录核验本组公开结果只需要 Python 标准库。准备好私有数据和参考训练记录后，可用同一入口训练指定种子。环境变量与缓存准备见[复现说明](../../REPRODUCING.md)。

```bash
python3 experiments/peak/run.py --verify
CUDA_VISIBLE_DEVICES=0 python experiments/peak/run.py --seed 3
```
