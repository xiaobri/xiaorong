# 全局专家角色消融：A–E 配对实验

本实验回答两个问题：当前全局专家的收益来自**帮助路由选择区域专家**，还是**直接提供分类预测**；在保留原模型的前提下，按冻结区域专家的训练表现重加权全局辅助损失，是否能稳定提高窗口级阳性 F1。

A–E 五组均已在种子 0、1、2、3、42 上完成 20 轮训练，共 25 次拟合。按各组在验证集选定的轮次和阈值，E 相对重训基线 D 平均变化 **+0.0006 个 F1 点**。另对 D、E 的十个选定检查点进行了原始 EEG 重放。[五组逐种子主表与解释](results/) · [25 条结果及混淆计数](results/per_seed.csv) · [D、E 原始 EEG 重放](results/raw_de.csv)

## 先核对现有模型

实现基于仓库的 [原训练器](../../source/I2MOE6_F1/train_eeg19_i2moe.py)、[模型](../../source/I2MOE6_F1/i2moe_eeg/model.py)与[损失](../../source/I2MOE6_F1/i2moe_eeg/losses.py)。五个冻结的区域蒸馏专家各自接收同一个 `[B,19,T]` 输入；本实验不新增脑区导联遮挡。全局专家（代码中的 `redundancy_expert`）融合五个冻结专家的 128 维特征。原路由器接收五个区域特征及 `detach()` 后的全局特征，输出六个稠密 softmax 权重，以这些权重融合专家的二分类 logits。

“区域”描述冻结检查点的来源与训练设定，不能仅凭名称认定当前推理只观察某个脑区。“全局冗余专家”也是代码名称；本实验不估计 PID 的冗余、独有或协同信息。

仓库既有 [基线结果](../../controls/baseline/results.csv)和[复现记录](../../REPRODUCING.md)是历史审计依据。**D 必须在本次相同输入、版本和配对协议下重训，不能把历史最佳值或历史五种子均值直接填入 D。**

## A–E 的精确定义

全部组保持相同网络尺寸、初始化方式、训练预算、冻结检查点和输入。下表的“全局特征参与路由”指送入 router 的第六个特征 token；“全局预测参与分类”指第六个专家的 logits 可以获得非零融合权重。

| 组 | 全局特征参与路由 | 全局预测参与最终分类 | 全局辅助损失 | 要回答的问题 |
|---|---|---|---|---|
| A | 否；第六个 token 置零 | 否；只融合五个区域预测 | 原辅助 focal | 五个区域预测经原尺寸路由器融合的控制组 |
| B | 是；使用 detached 全局特征 | 否；只融合五个区域预测 | 原辅助 focal | 全局特征是否帮助路由 |
| C | 否；第六个 token 置零 | 是；六专家融合 | 原辅助 focal | 全局预测是否提供额外分类价值 |
| D | 是；与原模型一致 | 是；与原模型一致 | 原辅助 focal | 本次重训的完整原基线 |
| E | 与 D 相同 | 与 D 相同 | 仅替换为下述加权 focal | 按冻结区域表现分配全局辅助训练权重是否有效 |
| E_mean（可选） | 与 D 相同 | 与 D 相同 | 将 E 的 `max` 换成 `mean` | E 的效果是否依赖“最佳区域专家”这一选择 |

A/B 在 **softmax 之前**把第六个路由 logit 设为负无穷，再对其余五个候选归一化；不是在 softmax 之后清零而不归一化。A/C 保留 `[B,6,128]` 路由输入尺寸，只把第六个 token 置零，不缩小路由器。

**A/B 仍运行并训练全局分支的原辅助任务和一致性项。** A 的全局分支不影响最终输出，只作为训练计算和随机数消耗的控制；B 的全局特征进入路由时已 detach，全局分支依靠辅助任务与一致性项学习。这使各组保留相同的分支调用、张量尺寸和随机扰动路径，避免简单删除模块同时改变初始化、dropout 或噪声消耗。相同种子不意味着不同组的参数更新与最终权重相同，也不保证跨硬件逐位复现。

相同参数量指分配的参数及其初始化相同；被置零或屏蔽的路径会有零梯度，所以不声称各组有效自由度完全相同，也不以这组控制实验比较推理效率。

### E 只改变一个已有损失项

对训练批中样本 `i`，令 `p_ir(y_i)` 为第 `r` 个冻结区域专家对**真实训练类别**的 softmax 概率，`r=1,...,5`：

```text
g_i = stopgrad(1 - max_r p_ir(y_i))
w_i = stopgrad((1 + g_i) / mean_batch(1 + g))

L_global_aux(E) = mean_i[w_i * Focal(global_logits_i, y_i)]
L_total = L_main + 0.05 * L_global_aux + 0.005 * L_consistency
```

`Focal` 表示原实现的逐样本 focal 项，`alpha=0.85, gamma=2`。A–D 使用未加权的 `L_global_aux`；E 保留原 `L_main`、原一致性项及其系数，只替换全局辅助 focal 的归约方式。没有新增损失分支、判别头、路由输入或推理步骤。批内权重均值为 1，用于避免整体辅助损失仅因权重尺度变化而增大。

`E_mean` 定义为 `g_i = stopgrad(1 - mean_r p_ir(y_i))`，其余完全相同。`max` 版本强调五个区域专家都对真实类不够确信的训练样本；这只是依赖概率校准的训练代理量，不能直接称为“真实能力缺口”。标签只用于训练损失权重；验证和部署的模型前向不读取真实标签，也不需要计算该权重。验证集上的标签条件统计只作事后诊断。

## 锁定训练与选择协议

| 项目 | 正式设置 |
|---|---|
| 种子 | `0, 1, 2, 3, 42`；同种子各组配对 |
| 训练 | 20 epochs；每轮 20,000 次抽样；共 400,000 次抽样 |
| 批量与采样 | 训练/验证 batch size 均为 128；`match_cam`；`ratio=30`；保留原尾批处理 |
| 优化 | AdamW；学习率 `0.0005`；weight decay `0.01`；FP32 |
| 主任务 | 原 focal，`alpha=0.85, gamma=2` |
| 辅助与一致性 | 全局辅助系数 `0.05`；原特征一致性系数 `0.005`；replacement std `1.0` |
| 全局融合网络 | hidden dim 128；4 heads；2 fusion layers；feedforward multiplier 2；dropout 0.3 |
| 路由网络 | hidden dim 128；2 layers；dropout 0.5；temperature 2.0 |
| 输入 | 原 19 导联预处理；target samples 2000；scale divisor `0.001`；frozen chunk size 0 |
| 阈值网格 | `0.01, 0.02, ..., 0.99`，共 99 点；阳性判定为严格 `p > threshold` |
| 选择规则 | 每轮选择验证 F1 最大的阈值，并列取较小阈值；跨轮并列取较晚轮次 |
| 主检查点 | 该组、该种子按上述规则得到的 `best_f1.pt`；不能换成 `best_auprc.pt` |

**主结果是每组、每个种子各自用同一验证集选 epoch 与阈值后得到的 F1，不是统一阈值，也不是独立测试集成绩。** 另报同一个 F1 选中检查点在固定阈值 `0.5` 下的指标；固定阈值结果不再另选 epoch。它能帮助观察阈值敏感性，但不消除检查点已在该验证集上选择造成的偏差。

阳性类别指标按 `F1 = 2TP / (2TP + FP + FN)` 计算，不能从跨种子平均 precision 与平均 recall 反算平均 F1。**机器报告中的 F1、precision、sensitivity（即 recall）、AP 和 Brier 使用原始 0–1 量纲，log loss 单位为 nats；配对差值沿用该指标的原量纲。** 本文和回报模板的展示表把 F1/precision/recall 乘以 100 转为百分制，F1 差值乘以 100 后称“百分点”。五种子汇总报告算术均值和样本标准差（`ddof=1`），比较使用同种子的配对差值；`ap` 是 average precision，不是梯形积分 PR-AUC。

本轮保留原清单划分。原清单只有 `file_path,target,mode`，不能证明患者独立；目录推断审计也不等于患者身份确认。完成角色诊断后，论文主结论仍需有明确患者标识的患者独立验证、锁定阈值后的独立测试，以及适当的外部数据验证。

## 环境与私有输入

### 无私有数据的代码检查

安装依赖后，从仓库根目录运行：

```bash
python -m unittest discover -s experiments/global_role_ablation/tests -v
python tools/verify_results.py
```

本 PR 已在 Windows、Python 3.13.3、PyTorch 2.6.0+cpu、torchvision 0.21.0+cpu 上通过 **41 项测试**，包括 D 与原基线的输出/损失/梯度逐位一致、路径干预、E 权重公式、六组真实训练循环的小型合成数据检查，以及 D/E 两轮中断恢复与连续训练的模型、优化器和 RNG 状态逐位一致。报告检查覆盖缺失运行、错误配置、NaN、采样记录、冻结预测、阈值/混淆计数与公开导出。仓库原有 13 组消融、2 组对照和 1,300 条逐轮记录的校验也通过。

这些是代码验证，不是 EEG 效果实验，也不替代下面的原 CUDA 环境复现。为了让原检查点保存逻辑支持 Windows，`atomic_torch_save` 仅把用于 `fsync` 的临时文件句柄改为可写、不截断模式；序列化和原子替换流程保持不变。

### 正式训练环境

所有命令从仓库根目录执行。原记录环境为 Python 3.10、PyTorch 2.4.0、torchvision 0.19.0、CUDA 12.1，依赖见 [requirements.txt](../../requirements.txt)。正式每个训练进程只使用一张可见 GPU；先核对 [原复现说明](../../REPRODUCING.md)中的私有路径与冻结检查点契约。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
export CZB_STORAGE="$PWD/private-data"
export CZB_RUNS="$PWD/private-runs"
export CZB_STORED_PREFIX="/eeg"
```

`CZB_STORED_PREFIX` 必须与授权清单 `file_path` 中需要映射掉的前缀一致，上例 `/eeg` 只是默认示例。`CZB_STORAGE` 下需有原五个冻结区域检查点及 EEG 数据。清单要求 `target` 为 0/1，`mode` 为 `Train`/`Val`；路径映射、输入预处理和冻结源身份继续由原训练器检查。迁移机器时可透传 `--csv-path`、`--data-root`、`--stored-prefix` 等原训练参数，五组必须使用同一配置。

### 先检查环境，再跑一个种子

有授权 EEG 和冻结检查点时，可先做一次真实输入前向/反向检查。检查模式使用独立输出根，不混入正式矩阵：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/global_role_ablation/run.py \
  --arm D --seed 3 --mode smoke \
  --output-root private-runs/global_role_smoke \
  --csv-path /authorized/manifest.csv \
  --data-root /authorized/eeg \
  --stored-prefix /eeg
```

`--mode debug` 默认为 1 epoch、2 个训练步和 256 个验证样本，可用于检查短训练流程；请另用如 `private-runs/global_role_debug` 的输出根。smoke/debug 不能提供正式效果结论，也不能接着在同一路径改成 formal：不同模式的运行身份会冲突。默认 `--mode formal` 锁定完整预算与结构。

先运行 D/seed3，确认数据、检查点、显存和完整验证流程能工作，再启动正式矩阵。下面的路径请替换为本机已授权位置。

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/global_role_ablation/run.py \
  --arm D --seed 3 --output-root private-runs/global_role_ablation \
  --csv-path /authorized/manifest.csv \
  --data-root /authorized/eeg \
  --stored-prefix /eeg
```

运行其他单组只改变 `--arm` 与 `--seed`。单组从已提交的 `resume.pt` 恢复时，在同一命令加 `--resume-run`；不要直接透传原训练器的 `--resume` 或 `--output-dir`。`run.py` 接受 `A,B,C,D,E,E_mean`；正式预算和结构参数由本实验锁定，不应为了改善某组结果单独调整。如果执行小样本检查、改变预处理或更换冻结源，必须另用输出根并在回报中明确标为非正式结果。设备、worker 数或确定性设置等原训练参数也必须在比较组间一致，汇总器会检查共同协议。

### 检查队列，再运行全部 25 次拟合

`suite.py` 默认生成 A–E × 五种子的完整矩阵，执行顺序为种子 `3,42,0,1,2`，每个种子依次执行 A–E；这不改变汇总的五种子集合。队列在单个可见 GPU 上串行执行。`--` 之后为透传给原训练器的公共参数；每组收到相同参数。dry-run 只打印计划，不加载 PyTorch、EEG 或检查点，也不验证这些私有输入是否可用。先检查输出及所有路径：

```bash
python experiments/global_role_ablation/suite.py \
  --output-root private-runs/global_role_ablation --dry-run -- \
  --csv-path /authorized/manifest.csv \
  --data-root /authorized/eeg \
  --stored-prefix /eeg
```

正式执行；如果前面的 D/seed3 已完成，使用恢复选项避免重复拟合：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/global_role_ablation/suite.py \
  --output-root private-runs/global_role_ablation --resume-existing -- \
  --csv-path /authorized/manifest.csv \
  --data-root /authorized/eeg \
  --stored-prefix /eeg
```

中断后在同一源码、协议和数据配置下重跑同一条 `--resume-existing` 命令。已有 `resume.pt` 的运行会进入原训练器的恢复/完成审计，已完成运行也会检查记录而非盲目跳过；失败运行保留各自的 `launcher.log`，队列遇到非零退出即停止。保留原输出与错误日志；不要通过删除协议文件、替换检查点或修改已有记录使不兼容的训练继续。

默认队列不包含 `E_mean`。若预先决定补做此敏感性对照，按五个种子完整补齐，并单独标注额外五次拟合：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/global_role_ablation/suite.py \
  --output-root private-runs/global_role_ablation \
  --arms E_mean --seeds 0 1 2 3 42 --resume-existing -- \
  --csv-path /authorized/manifest.csv \
  --data-root /authorized/eeg \
  --stored-prefix /eeg
```

不能只保留有利种子。如果在查看主实验结果后才决定运行 E_mean，应在回报中明确标为后续探索。

### 生成可回报的聚合结果

```bash
python experiments/global_role_ablation/report.py \
  --root private-runs/global_role_ablation \
  --output-dir outputs/global_role_ablation_report
```

输出包括 `per_seed.csv`、`summary.csv`、`paired_differences.csv`、`diagnostics.csv`、`report.json` 和 `RETURN_RESULTS.md`。默认汇总严格要求完整的 25 次正式拟合，每次 20 轮；缺失运行、预测归档或协议不一致会报错，不能把错误当作零分填表。失败时回报脱敏错误摘要并补齐运行，再生成正式报告。

包含可选 E_mean 的报告需在同一条命令追加 `--arms A B C D E E_mean`。需要阶段性回报时，可用 `--seeds 3` 等明确选择已经完整跑完全部组的种子，但必须标为部分矩阵，不能写成“五种子验证完成”。根据 [RETURN_TEMPLATE.md](RETURN_TEMPLATE.md)补充执行环境和解释。

## 保存什么，返回什么

每次拟合位于 `private-runs/global_role_ablation/<arm>/seed<seed>/`。保留原训练器的参数、训练逐轮记录、检查点、恢复状态和每轮 `predictions_epochXX.npz`，以及本实验的 `launch_contract.json`、`protocol.json`、`initialization.json`、`sampling_plan.json`、`sampling_epochXX.json` 和 `validation_rows.json`。`protocol.json` 记录源码与共同协议身份；初始化和抽样摘要用于核对同种子配对；有序验证行记录和预测用于在本机核对各组是否对同一批窗口进行评价。相关私有记录可能含路径或行标识，不直接作为公开附件。

公开回报限于聚合报告及去除私有信息的失败摘要。**不提交 EEG、检查点、原始 NPZ、验证行 ID、患者 ID、私有清单或机器绝对路径。** 原始日志若含这些信息，先生成脱敏摘要；不要把整个运行目录打包推送。

必须返回：

1. A–E 的完整 25 行逐种子主结果：选中 epoch、阈值、TP/FP/FN/TN、F1、precision、recall、AUPRC，以及同检查点固定 0.5 的结果；可选 E_mean 另列五行。
2. 各组五种子均值与样本标准差；`B-A`、`C-A`、`D-A`、`D-B`、`D-C`、`E-D` 的同种子差值及其汇总。报告每个种子的变化，不只报告均值或最好一次。
3. `diagnostics.csv` 中的全局专家、区域专家与路由诊断，并说明使用的阈值和分组定义；样本量不足或无法计算的字段如实标记。
4. 源码 commit、实际环境、完成数/失败数、协议核对结果、数据划分审计状态，以及去除私有路径的失败原因。

### 诊断字段的解释

`diagnostics.csv` 的 `repair_damage` 行将同种子的 A 作为参考，分别在各自验证所选阈值和共同固定 0.5 下计算：A 错而当前组对的 repair 数、A 对而当前组错的 damage 数，以及按全体窗口或 A 的错/对窗口归一化的比例。**这里的 E repair/damage 也是相对 A；E−D 的主指标比较看配对表，不把相对 A 的计数直接说成 E 修复 D。**

专家的“正确”统一用各自概率严格大于 0.5 判定。`top_selected` 是路由最大权重专家（并列取首个），不是事后最优专家。相关字段分别记录：至少一个区域专家正确；最大权重专家错但某区域/可参与融合专家正确；最大权重专家对而最终融合错。它们是诊断比例，不是最终分类 F1 的上界。全局专家单独的固定 0.5 F1、验证网格 F1、AP、Brier 和 log loss 即使在 A/B 中也会记录；A/B 的全局单独成绩不属于其最终输出成绩。

`label_assisted_bin` 按冻结五区域专家在 0.5 下的判断分为 `all_regions_wrong_at_0_5`、`mixed_regional_correctness_at_0_5`、`all_regions_correct_at_0_5`。每组报告样本量及当前模型在本组主阈值下的指标。这是使用验证标签的回顾性分组，**不是 E 的训练权重、可部署路由规则、真实能力缺口或 F1 上界**；无阳性等无法计算的指标按输出标记解释。

## 如何解释结果并决定下一步

| 主要比较 | 能支持的初步结论 | 仍需控制的解释 |
|---|---|---|
| B−A | 在没有全局预测参与分类时，输入全局特征是否改善路由 | 不能仅据增益解释为临床语义理解 |
| C−A | 不向路由提供全局 token 时，增加全局分类候选是否有益 | C 仍有六个候选，A 为五个；不要称为等候选数的纯信息增益 |
| D−B | 已有全局路由输入时，全局分类候选是否进一步有益 | 两组分别训练，新增主任务梯度也会改变全局专家 |
| D−C | 已有全局分类候选时，全局路由输入是否进一步有益 | 观察是否稳定，不能只看路由平均权重 |
| E−D | 只重加权全局辅助 focal 是否改进原基线 | 与固定 0.5、AUPRC及各种子共同判断，防止把阈值适配当完整机制证据 |
| E−E_mean（可选） | 最佳区域概率与平均区域概率的代理量选择是否重要 | 如果两者相近，不支持 `max` 机制具有独特优势 |

A–D 是训练干预的四组配对实验，差值不能直接相加作为严格的因果贡献或 PID 分解。即使 E 稳定优于 D，也先写成“这一辅助权重在本协议下有效”；要声称全局专家补充了区域专家，应进一步展示增益发生在哪类样本、是否有患者层面复现，以及是否优于相同信息与预算的简单替代。若 E 没有稳定增益，完整回报负结果，不扩大搜索后只展示成功变体。
