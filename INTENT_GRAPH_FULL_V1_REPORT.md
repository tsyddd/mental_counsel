# MindTrace 意图识别主对比：事件→意图图谱 vs 完整会话

主方法读取实际生产 Worker 维护的活跃意图节点，不再让模型读取 Top10 事件生成答案。完整历史基线读取截至当前用户发言的全部带时间、角色和会话边界的双方对话。

请求模型：`gpt-6-sol`。实际图谱版本：current deployed two-layer memory; scene/activity retrieval changes not yet integrated into production。

| 范围 | 已评分/预定节点 | 图谱 Recall | 完整历史 Recall | 图谱 Precision | 完整历史 Precision | 图谱 F1 | 完整历史 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 长会话 | 15/60 | 80.00% | 96.67% | 80.00% | 96.67% | 80.00% | 96.67% |
| 跨会话 | 7/60 | 66.67% | 100.00% | 75.00% | 100.00% | 70.59% | 100.00% |
| 全部 | 22/120 | 76.92% | 97.44% | 78.95% | 97.44% | 77.92% | 97.44% |

回放/预测失败记录 54，评分错误 44。已核对 1660 条检查点事件来源。

上表只计算成功评分节点，完整汇总另列所有预定 gold 分母的 Recall 下界。存在失败或缺失时不能把成功子集当作完整成绩。

本批为复用开发素材的合成诊断集，两个会话模式和两个节点配对依赖；尚未独立人工审核。两边基础模型相同，但累计调用预算不同。候选识别不等于用户确认或临床效果。

[评测协议](docs/intent-main-graph-vs-full-v1.md) · [冻结数据](data/recall_dataset_v1.json) · [完整结果](data/graph_full_v1_results.json) · [汇总与错误](data/graph_full_v1_summary.json) · [基线及评分调用](data/graph_full_v1_calls.jsonl)
