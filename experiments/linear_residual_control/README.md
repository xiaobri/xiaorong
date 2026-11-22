# 同一组 195 维 EEG 特征能否用线性残差读出？

在五脑区冻结专家与全局冗余专家的原始模型上，本实验比较四种点监督读出与两个历史配对校正。主要指标是协和固定验证划分的阳性窗口级 F1，百分制。五个种子共用同一份验证数据。开发种子为 3、42，追加种子为 0、1、2。实现从 [c1574bb](https://github.com/xiaobri/xiaorong/tree/c1574bb13bd805244f034d9d497b30adf99922ea) 分出，实验问题与预算见 [issue #1](https://github.com/xiaobri/xiaorong/issues/1)。

最佳线性头（合格采集组＋Focal）五种子 F1 为 **71.015 ± 0.172**，比基线高 **1.290** 点；同一次原始重放的历史配对＋刷新为 **72.120 ± 0.210**。[五种子结果、阈值与解释](RESULTS.md)

## 干预与比较口径

| 线性读出 | 头部目标 | 参与头部更新的 Train 窗口 |
| --- | --- | --- |
| [linear_focal_all](linear_focal_all/) | 原 Focal，阳性 α=0.85，γ=2 | 全部 |
| [linear_bce_all](linear_bce_all/) | 无权重的二元交叉熵 | 全部 |
| [linear_focal_eligible](linear_focal_eligible/) | 同配置 Focal | Train 清单中阳性、阴性均至少 50 个窗口的采集组 |
| [linear_bce_eligible](linear_bce_eligible/) | 无权重的二元交叉熵 | 同一 Train 支持集 |

每个脑区依次提供 32 个 pooled hidden coordinates、整窗脑区 logit margin，以及由八个块的 margin 得到的均值、总体标准差、最大值、最小值、最大两个值的均值和相邻差绝对值的均值。五区共 195 维。训练、推理直接调用历史分支的特征实现，按相同次序拼接，分离梯度并裁剪到 [−32, 32]。原始 EEG 重放注册相同的 block hooks。组 ID 只用于 Train 过滤。[原始特征定义](../../source/I2MOE/test333/fair_analytic_regions_20261120/analytic_models.py) · [新读出](study.py)

读出使用零初始化的 195 个权重和一个偏置。训练用同一次核心前向输出的 logits 作为偏移，加入 (−δ/2, +δ/2)，其中 δ 是单个线性输出。头部使用独立的 AdamW，权重衰减 0.01，偏置衰减 0，常数学习率。合格样本数为零时整次头部更新跳过。所有头共享相同的核心前向，梯度在输入特征与核心 logits 两处截断。核心沿用基线完整损失、AdamW、学习率 0.0005、batch size 128、每轮 20,000 抽样与 20 轮训练。每个种子逐轮核对全部训练抽样顺序、初始参数及五个冻结检查点。两种历史对照为单组件实验的 pairs 和 pairs_refresh。[历史分支与文献归属](../../REFERENCES.md) · [历史单组件定义](../../source/I2MOE/test333/fair_single_components/single.py)

每个开发种子训练四个头各三个学习率（0.0001、0.0005、0.002）。每种设置在每一轮使用原始 99 点阈值网格，严格执行 p > threshold，并在同轮 F1 并列时取先出现的较小阈值。跨轮并列取较晚轮次。每个 arm 选择两个开发种子的最佳 F1 的算术平均最高的学习率，使用整数混淆计数的有理数做比较，平局取较小学习率。写入含 40 个逐轮记录摘要的锁文件后，方可启动追加种子的 12 次拟合。各行分别选择自己的轮次和阈值。[学习率选择](selection.py)

最终在 166,079 个原始验证窗口（阳性 3,471、阴性 162,608）上重放已锁定检查点和阈值。三个参考也重新导出逐窗口分数。共同冻结卷积只计算一次，每个分片首末原生 batch 对照独立端到端前向。新头的检查点身份为 SHA256(核心文件的十六进制 SHA256 + 该头命名张量的十六进制 SHA256)，参考模型使用原检查点的文件 SHA256。精确阈值诊断固定同一个 99 点选中的检查点，全阴、全阳端点和全部唯一分数之间的切分均纳入，分数相同的行整组移动。精确 F1 并列取最大的切分阈值。主表、诊断表均比较双精度转换后的概率。[原始 EEG 导出](raw_scores.py) · [阈值实现](metrics.py)

## 计算与实现检查

每个种子同时运行一份独立 core，恢复相同随机状态、批次顺序和训练模式，逐轮比对全部可训练状态。第一轮还独立拟合一个头，对照共享运行的全部头参数、AdamW 一二阶矩及步数。在两份核心运行之间恢复 Python、NumPy、CPU 与 GPU RNG。另有真实 EEG 的 128 窗口特征对照，以及 CPU 梯度隔离、恢复后下一步更新、空 mask 和阈值并列测试。[真实特征检查](probe.py) · [CPU 检查](test_control.py)

每个头有 196 个参数与 394 个 AdamW 状态标量（392 个一二阶矩、2 个 step）。所有头都计入有限调参成本。每个种子的 400,000 个核心 Train 抽样供多个头同时使用；另有 400,000 个配对审计抽样重复处理同一序列。审计的第一轮也更新一个独立头。头部时间为 CUDA 同步后的逐头更新耗时。每个种子的共享核心训练时间、整轮训练加验证和审计时间、进程显存高水位在同一种子的多行中重复，只按种子计一次。训练成本不含原始 EEG 重放，逐轮完整耗时取原训练器计时，不含该轮结束后的核心检查点落盘。训练按不同配置的实际执行分配记录，历史方法的整个开发搜索预算未知。

## 复现

训练及原始 EEG 重放使用与基线一致的 Python/PyTorch 环境、私有 EEG、五个冻结专家、原始清单、冻结特征缓存、追加种子的原生尾批缓存与参考检查点。环境路径变量及上游缓存布局见[仓库复现说明](../../REPRODUCING.md)。在仓库根目录激活原环境，配置 CZB_STORAGE、CZB_RUNS、CZB_STORED_PREFIX；可通过 CZB_LINEAR_OUTPUT 指定独立的私有输出根目录。GPU 编号是本实验的任务分配，可按机器情况映射。一条命令启动一个阶段；参考重放可以在开发训练期间独立执行。

```bash
D=experiments/linear_residual_control
python -m unittest discover -s "$D" -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES=4 python "$D/probe.py"
python "$D/pipeline.py" --phase development
python "$D/pipeline.py" --phase references
python "$D/pipeline.py" --phase after-development
python "$D/report.py"
python "$D/verify.py"
python tools/verify_results.py
```

实际训练和重放的最小入口如下。seed 0、1、2 的入口会读取已锁定的学习率。中断后在原私有输出位置重新执行同一个 seed，原训练器以 resume.pt 为提交标记，头部从同一轮的原子快照恢复权重、AdamW 及配置；逐轮记录保留已提交的选模历史。

```bash
CUDA_VISIBLE_DEVICES=0 python "$D/train.py" --seed 3
CUDA_VISIBLE_DEVICES=1 python "$D/train.py" --seed 42
python "$D/selection.py"
CUDA_VISIBLE_DEVICES=1 python "$D/train.py" --seed 0
CUDA_VISIBLE_DEVICES=2 python "$D/raw_scores.py" --stage references --shard 0 --shards 6
```

普通电脑只需 Python 3 标准库和公开的 CSV，即可运行 `python experiments/linear_residual_control/verify.py`。此命令核验计数、F1、精确扫描的报告值、各阶段均值、样本标准差、逐种子配对差、选择和训练预算。重新搜索逐窗口精确阈值需要私有原始分数。
