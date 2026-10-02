# 双面观测架构 —— context-inspector × 上下文诊断工具 合并设计注记

> 版本：v1.0（2026-10-02）
> 依据：《06-上下文现状与标准对照评估》G-6/G-7 发现（回喂轮 wire 不落链、G10/G11 变换链上不可见）；C-6 裁定（诊断一期零侵入）；实施文档 v3.3（inspector 插件形态）。
> 性质：**合并权衡的结论档**——诊断一期落在本仓（evorule-llm-context-inspector），与 wire 面观测共用基础设施。01 主档讨论日志的同步由 owning 会话随动（本档为输入）。

## 一、两面分工（为什么合并进一个仓）

| 观测面 | 载体 | 数据源 | 覆盖 | 盲区 |
|---|---|---|---|---|
| **wire 面**（旁路） | inspector MITM（已常驻 19100/19101+） | 每轮 LLM 实收请求（旁路捕获，非记账） | **全部轮次**——含流式本地 ReAct 的中间态（server 无感知的那部分） | 上游真实响应原文（当前只记 usage/finish）；不经代理的调用 |
| **fact 面**（链上） | 诊断 CLI（本仓新增 `context-doctor`） | session fact WAL：Command(params.messages=首轮+audited 调用 wire)/PayloadUpdate(原始全文)/IoRequest/Stable/tool_trace | 事实面全量 + **首轮 wire** | **回喂轮 wire（G-6）**；G10/G11 变换（G-7） |

**合并的核心论点**：G-6 的盲区在 wire 面，G-6 的证据在 fact 面——单面都只见一半。诊断工具与 inspector 同仓后：

1. **基础设施复用**：tiktoken token 归因（tokens.rs）、canonical/哈希（canonical.rs + B 族原语）、fact WAL 读取（reconcile.rs 已打通 `read_wal_with_hash`）、provider 请求解析（provider/）——诊断零重复建设。
2. **对照检查成为一等公民**：`wire 轮次数 vs fact 可重建轮次数` 直接量化 G-6 盲区；wire 首轮 vs fact IoRequest 逐字段对照 = RL-C5（有损必有对照）的观测实现。
3. **两面语义严格分清**：wire 面是**旁路证据**（外部观测，非 RL-A2 记账权威）；fact 面是**链上权威**。O-214「记账先于压缩」要求的记账面收敛（回喂轮 wire 正式落链）仍是 evo-agent 侧的独立议题——旁路观测不替代记账，只补观测。

## 二、诊断一期范围（C-6 已裁定 + 06 档修正口径）

- 输入：fact WAL（必需）+ inspector WAL（可选，提供则出两面对照）+ 约束清单文件（可选，缺省启发提取）。
- 八层启发式归因（06 §3.5 修正口径）：wire 快照消息按格式特征归层（L1-L3 系统基底 / L4 goal（首轮 user）/ L5 tools / L6 记忆分区（## Stable Facts 等标记）/ L7 tool 角色 / L8 历史+hint/摘要）；事实面单独视图（__memory__ 原始/Stable sediment）。
- 确定性检查器：token 归因分布 / 工具膨胀（L7 占比+单条超标）/ 重复（跨消息内容哈希）/ 约束存在性+生存率（shadow 策略 #1，决策只记录不执行）/ I1·I2 字面级（否定式对冲检测）。
- 盲区显式标注：wire 轮次 > fact 可重建轮次 → G-6 量化报告（不虚构，不掩盖）。
- 输出：文本 + JSON 双格式（T-3 附加建议）；确定性自证（同输入两次运行逐字节一致）。
- shadow actions 落链形态待 T-4/R3——本地 `actions.jsonl` 暂存并显式标注「落链形态待定」。

## 三、不做的（红线边界）

- 不写事实层（RL-A1 只读）；不建独立诊断库（T-3 不落库，RL-B1/B5）；不动 LLM 出口（RL-B2）；语义级 I1/I2 留离线研究（RL-A3/RL-D1）；管理干预（auto）零涉及——shadow 只记录。
