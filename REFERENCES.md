# 论文与代码的对应关系

每个实验目录的 README 列出与该改动直接相关的文献。表中的「关系」说明具体借鉴到了哪一步。

| 论文 | 与本项目的关系 |
|---|---|
| [Tan 与 Le，ICML 2021，EfficientNetV2: Smaller Models and Faster Training](https://proceedings.mlr.press/v139/tan21a.html) | 共享 EEG 骨干的架构来源。冻结的五套权重、脑区划分及全局专家来自项目基线。 |
| [Lin 等，ICCV 2017，Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html) | 共享分类目标。若一组单独加入区域辅助 focal，该项目定义了它接入的位置、权重与系数。 |
| [Jia 等，ICML 2026，EmBrace: A Collective Knowledge Fusion Framework Toward Unified EEG Foundation Models](https://proceedings.mlr.press/v306/jia26d.html) | 研究动机层面的参考。EmBrace 在多种 EEG 基础模型之间融合同步后的中间表示。本项目使用原六专家，并自行设计区域统计、峰形和时间段读出。 |
| [Sagawa 等，ICLR 2020，Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731) | 按群组考察高风险的参考。类条件逆频率、上限 64 和风险移动平均的组合由项目实现。 |
| [Burges 等，ICML 2005，Learning to Rank Using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/) | 从分数差构造正负样本排序目标的参考。具体 EEG 配对规则、间隔与系数见对应实验。 |
| [Shrivastava 等，CVPR 2016，Training Region-Based Object Detectors with Online Hard Example Mining](https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html) | 当前批次困难样本挖掘的参考。项目的困难负例版本按分数截取前 16 个；风险匹配版本按区域风险距离取最近的 16 个。 |
| [Wang 等，CVPR 2020，Cross-Batch Memory for Embedding Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html) | 跨批次记忆的直接思想来源。原论文将先前的小批次表示用于度量学习。本项目存储先前负例，用于同采集组内的配对校正。 |
| [McDonnell 等，NeurIPS 2023，RanPAC: Random Projections and Pre-trained Models for Continual Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2793dc35e14003dd367684d93d236847-Abstract-Conference.html) | 冻结表示和累积统计量上学习读出器的思想参考。本项目累积的是区域配对差值对应的正则化方程。 |

同采集组风险分位负例选择、解析校正与双方分数刷新是项目中的组合。ICML 2026 的 EmBrace 支持考察 EEG 中间表示这一动机。配对分支的设计线索来自跨批次记忆和冻结表征上的统计读出。项目原有的出处说明见[组件归因附录](https://github.com/xiaobri/CZB-MOE/blob/5a2989f20d91e2f197241a7825ab60b1cc41e8a2/paper/component_attribution.tex)。
