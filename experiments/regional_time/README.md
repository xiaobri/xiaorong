# 五脑区时序重加权

将每个脑区末端的 63 个时间位置划分为八段，学习各区独立的归一化时间权重，经原冻结投影和分类器重新计算区域分数。仅区域专家的分数变化进入最终融合，全局专家使用基线分数。

八段划分、按脑区学习权重和经原分类器重打分是本项目实现。

## 协和验证结果

| 种子 | 选中轮次 | 锁定阈值 | 阳性窗口 F1（%） | 对同种子基线的增益（百分点） |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0.88 | 70.537 | +0.716 |
| 1 | 2 | 0.86 | 69.959 | +0.451 |
| 2 | 1 | 0.89 | 70.432 | +0.714 |
| 3 | 2 | 0.81 | 69.659 | +0.269 |
| 42 | 1 | 0.86 | 70.220 | +0.036 |

五种子平均 F1 为 **70.161**，种子间样本标准差为 0.357，相对基线的配对平均变化为 **+0.437 个百分点**。五个种子均在本轮从各自的基线初始化训练。

每个检查点的阈值按原 99 点网格选出。表中 F1 根据锁定阈值下 166,079 个原始 EEG 窗口的混淆矩阵计算。[逐种子混淆矩阵](results.csv) · [全部 100 条逐轮缓存验证记录](epochs.csv)

## 文献与实现来源

- 项目用此文作为利用 EEG 中间表示的研究背景。本文献提出的是多基础模型的表示融合。本实验在五个冻结脑区专家上构造自己的读出。 [Jia 等，ICML 2026，EmBrace: A Collective Knowledge Fusion Framework Toward Unified EEG Foundation Models](https://proceedings.mlr.press/v306/jia26d.html)
- 监督来自 focal loss，两类的权重和聚焦参数沿用基线。 [Lin 等，ICCV 2017，Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)

五脑区专家与全局冗余专家沿用项目基线。共用骨干和损失的来源见[文献索引](../../REFERENCES.md)。对应实现见 [single.py](../../source/I2MOE/test333/fair_single_components/single.py) · [allocation_models.py](../../source/I2MOE/test333/fair_expert_allocation_20261120/allocation_models.py) · [block_features.py](../../source/I2MOE/test333/fair_temporal_blocks_20261120/block_features.py)。

## 运行

从仓库根目录核验本组公开结果只需要 Python 标准库。准备好私有数据和参考训练记录后，可用同一入口训练指定种子。环境变量与缓存准备见[复现说明](../../REPRODUCING.md)。

```bash
python3 experiments/regional_time/run.py --verify
CUDA_VISIBLE_DEVICES=0 python experiments/regional_time/run.py --seed 3
```
