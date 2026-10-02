//! # evorule-llm-context-inspector
//!
//! LLM 上下文快照与健康评估审计链。
//!
//! 把 LLM agent 决策时的运行时上下文快照成结构化数据（请求侧
//! [`ContextSnapshot` 计划中的 model 模块] / 响应侧 `ResponseObservation`），
//! 经独立 BLAKE3 审计链落账，使「agent 决策时看到了什么」可验证、可回放、可审计。
//!
//! ## 模块规划（实施文档 v3 基线，随阶段落地）
//!
//! | 模块 | 职责 | 阶段 |
//! |------|------|------|
//! | `model` | ContextSnapshot / ResponseObservation / ContextSkeleton / Composition / UsageInfo | Phase 1 |
//! | `canonical` | canonical JSON 封装：整数纪律、BTreeMap 键序、禁 preserve_order 守护 | Phase 1 |
//! | `metrics` | 五维健康指标（压力 / cache / 漂移+三显式检测 / 截断 / 重复率），全万分比整数 | Phase 1-2 |
//! | `provider` | LlmProvider trait + OpenAI / Anthropic / DeepSeek / Ollama 四实现 + SSE | Phase 2 |
//! | `chain` | InspectorLog：append / WAL / recover / verify_chain / anchor_head | Phase 3 |
//! | `proxy` | in-process MITM 监听（SSE 零缓冲转发；19100 基址，loopback-only） | Phase 3 |
//! | `api` | REST handlers（由 evorule-server 挂载；/admin/bindings 等） | Phase 3 |
//!
//! ## 硬性纪律（红线摘要，全文见实施文档 §2）
//!
//! - **隐私默认**：只存结构与内容哈希（skeleton），原文仅 `--store-content`
//!   时旁挂文件存放且不入链；任何哈希输入不含 HTTP header（API 密钥安全承诺）。
//! - **确定性**：链哈希输入全整数无浮点；时间一律 epoch 毫秒 u64；
//!   canonical JSON 用 serde_json 默认 BTreeMap 键序，禁开 preserve_order。
//! - **哈希 SSOT**：内容哈希走 `evorule_hash`；链步复用
//!   `evorule_reactor::chain_step`（`blake3(prev + content)`，genesis=`"genesis"`）；
//!   禁自写 blake3，禁第二套链步实现。
//! - **G5 代码质量**：0 panic / unwrap / expect（workspace 同款 lints，deny 级）。
//!
//! 当前状态：Phase 0 骨架（设计超前于代码是常态，代码为唯一标准）。

/// 骨架冒烟测试：证明两条哈希依赖可链接、链步确定性成立。
///
/// - `evorule_hash::digest`：内容哈希原语（B 族唯一合法入口）。
/// - `evorule_reactor::chain_step`：链步单一真相源（冻结仓 re-export）。
#[cfg(test)]
mod smoke_tests {
    #[test]
    fn digest_returns_64hex() {
        let h = evorule_hash::digest(b"genesis");
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn chain_step_is_deterministic() {
        let a = evorule_reactor::chain_step("genesis", "aaa");
        let b = evorule_reactor::chain_step("genesis", "aaa");
        assert_eq!(a, b, "同输入必须同输出（确定性红线）");
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn chain_step_differs_on_different_content() {
        let a = evorule_reactor::chain_step("genesis", "aaa");
        let b = evorule_reactor::chain_step("genesis", "bbb");
        assert_ne!(a, b);
    }
}
