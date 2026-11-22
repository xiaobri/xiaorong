# 复现与数据字段

## 在普通电脑上核验公开结果

从仓库根目录执行。校验器只使用 Python 标准库。

```bash
python3 tools/verify_results.py
python3 experiments/pairs_refresh/run.py --verify
```

每个实验的 `results.csv` 有五行。阈值使用严格大于号，`tp/fp/fn/tn` 是锁定检查点和阈值后在原始 EEG 上得到的窗口计数。F1 的百分制计算式为 `200 × TP / (2 × TP + FP + FN)`。`delta_vs_baseline` 是同种子 F1 相减的百分点。`changed_cached_decisions` 统计原始 EEG 复核与训练时冻结特征缓存做出的判定差异。

`epochs.csv` 每组 100 行，包含五个种子各 20 轮的缓存验证计数、该轮选出的阈值、训练采样数、优化步数及批顺序的 SHA-256。跨轮选验证 F1 最高的检查点，并列时取靠后的轮次。汇总中的标准差为种子间样本标准差。开发种子为 3、42，追加种子为 0、1、2。

另附五个基线的核心初始化与采样顺序摘要，以及 65 个消融拟合的核心初始化和选中检查点摘要。校验器逐组比对基线记录。[基线配对证据](results/baseline_pairing.csv) · [拟合与检查点身份](results/fit_identities.csv)

每项实验的训练入口固定干预名称。所有入口调用共享的基线、采样器及经核对的组件实现。历史记录中 `moment_plain` 的 seed 3、42 来自 `plain`，`moment_recipe` 的 seed 3、42 来自 `risk_power`。对应入口自动分派到历史训练脚本；历史 `risk_power` 的显存分配上限为 32 GiB，本轮新训练为 4 GiB。训练批量、优化设置与评估口径一致。

## 私有输入与环境

正式训练使用 Python 3.10、PyTorch 2.4.0、torchvision 0.19.0 和 CUDA 12.1。每个训练进程只看见一张 GPU。准备具有访问授权的原始协和清单、EEG 数组、五个冻结脑区检查点及配对参考记录。公开复核只用聚合计数。清单位置、冻结专家相对路径和散列值见[上游数据契约](https://github.com/xiaobri/CZB-MOE/blob/5a2989f20d91e2f197241a7825ab60b1cc41e8a2/results/protocol.json)。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
export CZB_STORAGE="$PWD/private-data"
export CZB_RUNS="$PWD/private-runs"
# 按授权清单的 file_path 列设置 CZB_STORED_PREFIX。
export EXP="$PWD/source/I2MOE/test333"
export REP="$EXP/fair_additional_seeds"
export SINGLE="$EXP/fair_single_components"
```

`CZB_STORAGE` 存放清单、冻结专家、特征缓存、基线配置与本轮输出。`CZB_RUNS` 存放原严格基线的初始化/采样证明、完整模型参考训练和时间块缓存。正式训练读取私有参考配置和缓存中的路径与摘要。迁移到新机器时，用新的路径生成一致的清单、缓存和配对证明。初次重建原始 seed 3、42 的基线、缓存和完整模型可按[固定版本的上游准备步骤](https://github.com/xiaobri/CZB-MOE/blob/5a2989f20d91e2f197241a7825ab60b1cc41e8a2/REPRODUCING.md)执行，所需 Python 模块位于本仓库的 `source/`。

## 追加三个种子的配对参考

如果已经有同一协议的参考训练和缓存，直接使用其所在的存储根。初次准备时，先计算 0、1、2 的补充训练窗口和各轮原生尾批次，再从头训练三个基线对照。以下为顺序执行示例。

```bash
python "$REP/supplement.py"
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" python "$REP/supplement.py" --shard "$shard"
done
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python "$REP/train.py" --method baseline --seed "$seed"
done
```

若要重算此前完整模型的全部对照，可同样训练 `--method fresh_both` 的三个追加种子，再执行 `$REP/replay.py --study-root "$PWD" --shard N`（N 取 0 至 7，每个进程指定一张 GPU），最后运行 `python "$REP/report.py" --study-root "$PWD"`。该步骤还需要前述上游完整模型的开发种子运行记录。本仓库保留用于合并记录的两张上游历史聚合表。[开发种子选中记录](results/selected.csv) · [原始窗口历史复核](results/component_xiehe.csv)

## 单组重训、全组队列和原始 EEG 复核

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/pairs_refresh/run.py --seed 2
```

程序会检查原基线核心参数的初始化、冻结权重和语义配置，并逐轮核对采样顺序。请为重训使用独立的实验存储目录，保留所需的参考数据结构。运行常规回归测试可用以下命令。

```bash
(cd "$SINGLE" && python -m unittest discover -s tests -p 'test_*.py' -v)
```

用 `python "$SINGLE/queue_fits.py"` 在 GPU 0–7 上排入 61 项新训练；前述四项历史结果继续复用。五个种子的原始 EEG 复核与汇总可以按下列顺序执行。重算整套消融总表时，汇总器从追加种子参考训练目录读取基线和完整模型的 `seed_results.csv`。下例在尚未生成该文件时使用此次归档的十条对照记录。

```bash
reference_dir="$CZB_STORAGE/I2MOE_test333_fresh_both_additional_20261122"
if [ ! -f "$reference_dir/seed_results.csv" ]; then
  cp results/reference_seed_results.csv "$reference_dir/seed_results.csv"
fi
mkdir -p "$CZB_STORAGE/I2MOE_single_components_20261122/raw"
for seed in 3 42 0 1 2; do
  for shard in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$shard" python "$SINGLE/replay.py" --study "$PWD" --seed "$seed" --shard "$shard"
  done
done
python "$SINGLE/report.py" --study "$PWD"
```

原始 EEG 复核使用同一清单的全部 166,079 个验证窗口和训练时选定的阈值，检查每个已选检查点的身份和每批标签顺序。模型源码及主训练脚本与本次 61 项拟合时记录的 SHA-256 一致。基线依赖来自[固定版本 CZB-MOE](https://github.com/xiaobri/CZB-MOE/tree/5a2989f20d91e2f197241a7825ab60b1cc41e8a2)，历史 risk_power 启动脚本的机器目录由公开路径配置解析。

## 绘图

已生成的图见[逐种子增益图](results/changes.png)。重新绘图需要 Matplotlib 和 Noto Sans CJK 字体。

```bash
python -m pip install matplotlib
python "$SINGLE/plot.py" results/single_component_seeds.csv results/changes.png
```
