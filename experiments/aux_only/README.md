# 冻结脑区上的区域辅助损失

区域辅助项为五个区域分类器 focal 损失的平均，乘静态组权重和系数 0.1。此处直接作用于基线冻结区域的输出。

冻结输出对应的辅助项对可训练参数的梯度为零。结果中的微小重训差异不能归为该项的学习效果。区域目标及其系数为本项目设置。

## 协和验证结果

| 种子 | 选中轮次 | 锁定阈值 | 阳性窗口 F1（%） | 对同种子基线的增益（百分点） |
|---:|---:|---:|---:|---:|
| 0 | 9 | 0.81 | 69.821 | +0.000 |
| 1 | 14 | 0.79 | 69.508 | +0.000 |
| 2 | 13 | 0.81 | 69.718 | +0.000 |
| 3 | 2 | 0.81 | 69.436 | +0.047 |
| 42 | 5 | 0.82 | 70.220 | +0.036 |

五种子平均 F1 为 **69.741**，种子间样本标准差为 0.310，相对基线的配对平均变化为 **+0.017 个百分点**。五个种子均在本轮从各自的基线初始化训练。

每个检查点的阈值按原 99 点网格选出。表中 F1 根据锁定阈值下 166,079 个原始 EEG 窗口的混淆矩阵计算。[逐种子混淆矩阵](results.csv) · [全部 100 条逐轮缓存验证记录](epochs.csv)

## 文献与实现来源

- 监督来自 focal loss，两类的权重和聚焦参数沿用基线。 [Lin 等，ICCV 2017，Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)
- 五个被监督的区域编码器采用 EfficientNetV2-S 骨干。五脑区划分和六专家路由由原项目定义。 [Tan 与 Le，ICML 2021，EfficientNetV2: Smaller Models and Faster Training](https://proceedings.mlr.press/v139/tan21a.html)

五脑区专家与全局冗余专家沿用项目基线。共用骨干和损失的来源见[文献索引](../../REFERENCES.md)。对应实现见 [single.py](../../source/I2MOE/test333/fair_single_components/single.py) · [allocation_models.py](../../source/I2MOE/test333/fair_expert_allocation_20261120/allocation_models.py) · [test_single.py](../../source/I2MOE/test333/fair_single_components/tests/test_single.py)。

## 运行

从仓库根目录核验本组公开结果只需要 Python 标准库。准备好私有数据和参考训练记录后，可用同一入口训练指定种子。环境变量与缓存准备见[复现说明](../../REPRODUCING.md)。

```bash
python3 experiments/aux_only/run.py --verify
CUDA_VISIBLE_DEVICES=0 python experiments/aux_only/run.py --seed 3
```
