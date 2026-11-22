# 时序与全局残差＋原加权排序＋区域辅助

将每个脑区末端的 63 个时间位置划分为八段，学习各区独立的归一化时间权重，经原冻结投影和分类器重新计算区域分数。全局残差读取五区特征与时间权重的变化，用原六专家路由权重融合。配套训练采用类条件静态采集组权重（上限 64）与批内排序。排序为每个阳性选择区域风险最接近的至多 16 个负例。区域辅助项为五个区域分类器 focal 损失的平均，乘静态组权重和系数 0.1。

此处区域重加权层可学习，区域辅助项可向该层传递梯度。与 time_native 相减可估计辅助监督的条件增益。

## 协和验证结果

| 种子 | 选中轮次 | 锁定阈值 | 阳性窗口 F1（%） | 对同种子基线的增益（百分点） |
|---:|---:|---:|---:|---:|
| 0 | 8 | 0.93 | 71.155 | +1.334 |
| 1 | 2 | 0.91 | 70.780 | +1.271 |
| 2 | 3 | 0.85 | 70.790 | +1.071 |
| 3 | 2 | 0.87 | 71.017 | +1.627 |
| 42 | 7 | 0.93 | 70.962 | +0.778 |

五种子平均 F1 为 **70.941**，种子间样本标准差为 0.159，相对基线的配对平均变化为 **+1.216 个百分点**。五个种子均在本轮从各自的基线初始化训练。

每个检查点的阈值按原 99 点网格选出。表中 F1 根据锁定阈值下 166,079 个原始 EEG 窗口的混淆矩阵计算。[逐种子混淆矩阵](results.csv) · [全部 100 条逐轮缓存验证记录](epochs.csv)

## 文献与实现来源

- 项目用此文作为利用 EEG 中间表示的研究背景。本文献提出的是多基础模型的表示融合。本实验在五个冻结脑区专家上构造自己的读出。 [Jia 等，ICML 2026，EmBrace: A Collective Knowledge Fusion Framework Toward Unified EEG Foundation Models](https://proceedings.mlr.press/v306/jia26d.html)
- 对正负样本分数差使用平滑排序目标借鉴了 RankNet。具体间隔 0.5、损失系数 0.03 和配对规则由本项目设定。 [Burges 等，ICML 2005，Learning to Rank Using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
- 监督来自 focal loss，两类的权重和聚焦参数沿用基线。 [Lin 等，ICCV 2017，Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)

五脑区专家与全局冗余专家沿用项目基线。共用骨干和损失的来源见[文献索引](../../REFERENCES.md)。对应实现见 [single.py](../../source/I2MOE/test333/fair_single_components/single.py) · [allocation_models.py](../../source/I2MOE/test333/fair_expert_allocation_20261120/allocation_models.py) · [alignment_models.py](../../source/I2MOE/test333/fair_time_alignment_20261119/alignment_models.py)。

## 运行

从仓库根目录核验本组公开结果只需要 Python 标准库。准备好私有数据和参考训练记录后，可用同一入口训练指定种子。环境变量与缓存准备见[复现说明](../../REPRODUCING.md)。

```bash
python3 experiments/time_aux/run.py --verify
CUDA_VISIBLE_DEVICES=0 python experiments/time_aux/run.py --seed 3
```
