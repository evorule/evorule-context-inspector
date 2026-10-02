# evorule-llm-context-inspector

> LLM 上下文快照与健康评估审计链 —— 把 agent 决策时的上下文结构化入独立 BLAKE3 审计链，
> 让「agent 决策时看到了什么」可验证、可回放、可审计。

**状态**:🟡 Phase 0 骨架(设计超前于代码是常态,代码为唯一标准)
**执行基线**:`D:\knowledge\3-Topics\evorule-context-inspector-实施文档.md`(v3 基线,唯一执行标准)

## 这是什么

evorule 治理 LLM agent 的规则决策时,审计链记录了规则执行事实(触发哪条规则/命中什么 trace/因果链),
但 **agent 做决策时 context window 长什么样完全看不见**。本仓补上这条链:

- **请求侧** `ContextSnapshot`:LLM 调用发出前的上下文定格(组成 token 计数 + 结构骨架)
- **响应侧** `ResponseObservation`:回包后追加(真实 usage / finish_reason / 截断检测),按 `response_to` 配对
- **五维健康指标**:压力(四档左闭右开)/ cache 命中率 / 漂移(multiset Jaccard + SystemPromptChanged / ToolSetChanged / ContextReset 三显式检测)/ 截断 / 重复率(word-shingle),全万分比整数存储
- **独立 BLAKE3 审计链**:链步复用 `evorule-reactor::chain_step`,链头定期入 G-A1 Ed25519 锚点流
- **in-process 集成**:evorule-server 可选 feature(默认开,可关),MITM 监听随 server 启停,端口段 19100(loopback-only)

## 隐私模型(默认强隐私)

| 层 | 内容 | 默认存储 |
|----|------|---------|
| 每条内容哈希 | system / tool / message 原文的 BLAKE3 | 只存哈希,不存原文 |
| `skeleton_hash` | canonical JSON(结构骨架) | 恒存 —— **从存储结构即可重算,链验证强语义** |
| `content_hash` | 完整 messages+tools 原文哈希 | 仅 `--store-content` 时原文旁挂文件存放,**不入链** |

任何哈希输入均不含 HTTP header(`Authorization` / `x-api-key` 永不进哈希、日志与落盘)。

## 哈希依赖(B 族纪律)

- 内容哈希 → `evorule_hash::digest` / `json_digest`(禁自写 blake3)
- 链步 → `evorule_reactor::chain_step`(`blake3(prev + content)`,genesis=`"genesis"`),库依赖复用,不触碰规则执行路径
- canonical JSON → serde_json 默认 BTreeMap 键序,禁开 `preserve_order`
- 本仓哈希域登记于 `D:\evorule-hash\README.md` 哈希域注册表(Phase 1 落地)

## 阶段计划(实施文档 §4)

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | 骨架与 registry 登记 | 🚧 |
| 1 | 数据模型与哈希(canonical JSON + skeleton_hash + 域登记) | ☐ |
| 2 | Provider 与流式(4 provider + SSE) | ☐ |
| 3 | 链与 server 集成(InspectorLog + MITM + verify-chain 协同) | ☐ |
| 4 | console-cloud UI(Context/ 六组件) | ☐ |
| 5 | 文档与发布 | ☐ |

## 工程纪律

- G5 代码质量:0 panic / unwrap / expect(lints deny 级,测试模块除外)
- 测试为唯一证据:验收矩阵 20 条全带测试路径;覆盖率 gate(Windows cargo-llvm-cov / CI Linux tarpaulin)
- 每阶段 DoD:fmt / clippy 净 + test 全绿 + 覆盖率 gate + 密扫 PASS + commit

## 许可

**AGPL-3.0-or-later** —— 见 [LICENSE](LICENSE) / [NOTICE.md](NOTICE.md);
双许可决策指引见 [DUAL_LICENSE.md](DUAL_LICENSE.md)。
