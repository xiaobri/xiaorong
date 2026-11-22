# 采集组风险加权

直接对原六专家融合分数的 focal 损失按类和采集组加权。每步根据历史组风险调整权重，保留原来的全局专家目标和交互损失。

该干预测量重加权直接作用于原模型时的效果。

## 协和验证结果

| 种子 | 选中轮次 | 锁定阈值 | 阳性窗口 F1（%） | 对同种子基线的增益（百分点） |
|---:|---:|---:|---:|---:|
| 0 | 10 | 0.80 | 70.308 | +0.487 |
| 1 | 8 | 0.74 | 69.835 | +0.326 |
| 2 | 16 | 0.73 | 69.404 | -0.314 |
| 3 | 16 | 0.81 | 70.002 | +0.612 |
| 42 | 11 | 0.73 | 70.388 | +0.204 |

五种子平均 F1 为 **69.987**，种子间样本标准差为 0.396，相对基线的配对平均变化为 **+0.263 个百分点**。五个种子均在本轮从各自的基线初始化训练。

每个检查点的阈值按原 99 点网格选出。表中 F1 根据锁定阈值下 166,079 个原始 EEG 窗口的混淆矩阵计算。[逐种子混淆矩阵](results.csv) · [全部 100 条逐轮缓存验证记录](epochs.csv)

## 文献与实现来源

- 按采集组强调高风险样本借鉴了 Group DRO 的问题设定。实现采用项目定义的类条件逆频率、64 的权重上限以及组风险移动平均。 [Sagawa 等，ICLR 2020，Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731)
- 监督来自 focal loss，两类的权重和聚焦参数沿用基线。 [Lin 等，ICCV 2017，Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)

五脑区专家与全局冗余专家沿用项目基线。共用骨干和损失的来源见[文献索引](../../REFERENCES.md)。对应实现见 [single.py](../../source/I2MOE/test333/fair_single_components/single.py) · [cohort_models.py](../../source/I2MOE/test333/fair_cohorts_20261115/cohort_models.py) · [complement_execution.py](../../source/I2MOE/test333/fair_complement_20261118/complement_execution.py)。

## 运行

从仓库根目录核验本组公开结果只需要 Python 标准库。准备好私有数据和参考训练记录后，可用同一入口训练指定种子。环境变量与缓存准备见[复现说明](../../REPRODUCING.md)。

```bash
python3 experiments/weight/run.py --verify
CUDA_VISIBLE_DEVICES=0 python experiments/weight/run.py --seed 3
```
