//! # evorule-llm-context-inspector
//!
//! LLM 上下文快照与健康评估审计链。
//!
//! 把 LLM agent 决策时的运行时上下文快照成结构化数据（请求侧
//! `ContextSnapshot` / 响应侧 `ResponseObservation`），
//! 经独立 BLAKE3 审计链落账，使「agent 决策时看到了什么」可验证、可回放、可审计。
//!
//! ## 模块（实施文档 v3.1 基线）
//!
//! | 模块 | 职责 | 状态 |
//! |------|------|------|
//! | `model` | ContextSnapshot / ResponseObservation / ContextSkeleton / Composition / UsageInfo + 核账与回填 | ✅ Phase 1 |
//! | `canonical` | canonical JSON 封装：整数纪律、BTreeMap 键序、禁 preserve_order 守护 | ✅ Phase 1 |
//! | `tokens` | token 估算（tiktoken 绑定，不自研分词）+ 图片 token 各家计法 | ✅ Phase 2 |
//! | `provider` | LlmProvider trait + OpenAI / Anthropic / DeepSeek / Ollama 四实现 + StreamAggregator | ✅ Phase 2 |
//! | `metrics` | 五维健康指标（压力 / cache / 漂移+三显式检测 / 截断 / 重复率），全万分比整数 | ✅ Phase 2-3 |
//! | `chain` | InspectorLog：append / WAL 先写 / recover / verify_chain / anchor_head | ✅ Phase 3 |
//! | `proxy` | 插件进程内 MITM 监听（SSE 零缓冲转发；19100 基址，loopback-only） | ✅ Phase 3 |
//! | `api` | REST handlers（admin 口挂载；/admin/bindings、查询面） | ✅ Phase 3（cli/锚点扩展下一增量） |
//!
//! ## 硬性纪律（红线摘要，全文见实施文档 §2）
//!
//! - **隐私默认**：只存结构与内容哈希（skeleton），原文仅 `--store-content`
//!   时旁挂文件存放且不入链；任何哈希输入不含 HTTP header（API 密钥安全承诺）。
//! - **确定性**：链哈希输入全整数无浮点（`canonical` 强制拒收浮点）；
//!   时间一律 epoch 毫秒 u64；canonical JSON 用 serde_json 默认 BTreeMap 键序，
//!   禁开 preserve_order（守护测试钉死）。
//! - **哈希 SSOT**：内容哈希走 `evorule_hash`；链步复用
//!   `evorule_reactor::chain_step`（`blake3(prev + content)`，genesis=`"genesis"`）；
//!   禁自写 blake3，禁第二套链步实现。
//! - **G5 代码质量**：0 panic / unwrap / expect（deny 级 lints，测试模块除外）。

pub mod anchor;
pub mod api;
pub mod canonical;
pub mod chain;
pub mod metrics;
pub mod model;
pub mod provider;
pub mod proxy;
pub mod reconcile;
pub mod tokens;

use uuid::Uuid;

/// 与 evorule 生态对齐的哈希形态：64 字符 hex 字符串（reactor `hash.rs` 同款，无 `blake3:` 前缀）。
/// 会话锚/审计链 64-hex 无前缀；禁与存储字段的 `blake3:` 前缀形态混用（02 总表 B 族）。
pub type HashHex = String;

/// 会话 ID：直接 re-export evorule-governance 的定义（`pub type SessionId = u64`），
/// 不自定义第二来源，避免双源漂移（设计稿 §2.1）。
pub use evorule_governance::SessionId;

/// 事实 ID：re-export evorule-reactor 的定义（`pub struct FactId(pub u64)`）。
/// 序列化口径与主链 `fact_to_stable_json` 一致 = 裸 JSON 数字（见 `model::SessionLink`）。
pub use evorule_reactor::FactId;

/// LLM provider 种类。序列化为稳定小写字符串（骨架 `provider` 字段用其 `as_str` 形态）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    OpenAi,
    Anthropic,
    DeepSeek,
    Ollama,
    /// 自定义协议接入（配置声明），窗口上限未知时 pressure 置 None（§2.6）。
    Custom,
}

impl ProviderKind {
    /// 骨架内使用的稳定字符串形态（跨版本不变；新增变体只能追加不能改写既有值）。
    pub fn as_str(self) -> &'static str {
        match self {
            ProviderKind::OpenAi => "openai",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::DeepSeek => "deepseek",
            ProviderKind::Ollama => "ollama",
            ProviderKind::Custom => "custom",
        }
    }
}

/// 单条消息的内容形态（结构信息，原文永不入模型 — 只入哈希）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentKind {
    Text,
    Image,
    Mixed,
    ToolCall,
    ToolResult,
}

/// token 计数来源。估算值仅请求瞬间暂态；响应回填后升级为 ProviderReported，
/// 压力与账目以 ProviderReported 为准（设计稿 §2.3 回填规则）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenSource {
    Estimated,
    ProviderReported,
}

/// 会话关联途径（三级机制，§4.2）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingKind {
    /// agent 请求带 `X-Evorule-Session` header。
    Header,
    /// 逐会话独立监听端口绑定（19101 起递增，默认推荐）。
    PortBinding,
    /// 未识别流量，进 unattributed 桶（产出 `ContextSessionUnknown` issue）。
    Unattributed,
}

/// 本仓统一错误类型（类型化，不用 String；R4 fail-fast：校验失败明确报错拒收）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InspectorError {
    /// canonical JSON 输入含浮点（链哈希输入整数纪律，§2.7）。
    FloatForbidden { path: String },
    /// Composition 核账失败：显式 total 与分项之和不一致（追加时拒收，§2.2）。
    TokenAccountMismatch { computed: u64, provided: u64 },
    /// Composition 分项求和溢出 u64。
    TokenOverflow,
    /// Observation 与 Snapshot 配对失败（`response_to` 不等于 `snapshot_id`）。
    PairingMismatch {
        snapshot_id: Uuid,
        observation_id: Uuid,
    },
    /// 字段级一致性校验失败（如 snapshot.model 与 skeleton.model 不一致）。
    InconsistentField { field: &'static str },
    /// 输入非法（空字段 / 哈希格式不符等）。
    InvalidInput(String),
    /// WAL / 文件 I/O 失败（write-ahead 失败 = 拒收且内存不变，R4）。
    Io(String),
    /// WAL 损坏或被篡改：fail-fast 拒载并定位记录序号（验收 #12）。
    WalCorrupt { seq: u64, reason: String },
}

impl core::fmt::Display for InspectorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InspectorError::FloatForbidden { path } => {
                write!(
                    f,
                    "canonical JSON 拒收浮点（路径 {path}）：链哈希输入只允许整数与字符串"
                )
            }
            InspectorError::TokenAccountMismatch { computed, provided } => {
                write!(
                    f,
                    "token 核账失败：分项之和 {computed} ≠ 显式提供 {provided}，拒收"
                )
            }
            InspectorError::TokenOverflow => write!(f, "token 分项求和溢出 u64"),
            InspectorError::PairingMismatch {
                snapshot_id,
                observation_id,
            } => {
                write!(f, "配对失败：observation {observation_id} 的 response_to ≠ snapshot {snapshot_id}")
            }
            InspectorError::InconsistentField { field } => {
                write!(f, "字段不一致：{field}")
            }
            InspectorError::InvalidInput(msg) => write!(f, "非法输入：{msg}"),
            InspectorError::Io(msg) => write!(f, "WAL/文件 I/O 失败：{msg}"),
            InspectorError::WalCorrupt { seq, reason } => {
                write!(f, "WAL 损坏（记录序号 {seq}）：{reason}——fail-fast 拒载")
            }
        }
    }
}

impl std::error::Error for InspectorError {}

#[cfg(test)]
mod type_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn provider_kind_wire_forms_are_stable() {
        // 稳定字符串契约：骨架 provider 字段与 serde 线形态跨版本不变（只增不改）。
        assert_eq!(ProviderKind::OpenAi.as_str(), "openai");
        assert_eq!(ProviderKind::Anthropic.as_str(), "anthropic");
        assert_eq!(ProviderKind::DeepSeek.as_str(), "deepseek");
        assert_eq!(ProviderKind::Ollama.as_str(), "ollama");
        assert_eq!(ProviderKind::Custom.as_str(), "custom");
        for (k, wire) in [
            (ProviderKind::OpenAi, "openai"),
            (ProviderKind::Anthropic, "anthropic"),
            (ProviderKind::DeepSeek, "deepseek"),
            (ProviderKind::Ollama, "ollama"),
            (ProviderKind::Custom, "custom"),
        ] {
            assert_eq!(serde_json::to_string(&k).unwrap(), format!("\"{wire}\""));
        }
    }

    #[test]
    fn enum_wire_forms_are_stable() {
        assert_eq!(
            serde_json::to_string(&BindingKind::PortBinding).unwrap(),
            "\"port_binding\""
        );
        assert_eq!(
            serde_json::to_string(&TokenSource::ProviderReported).unwrap(),
            "\"provider_reported\""
        );
        assert_eq!(
            serde_json::to_string(&ContentKind::ToolCall).unwrap(),
            "\"tool_call\""
        );
    }

    #[test]
    fn error_display_covers_all_variants() {
        let errs = vec![
            InspectorError::FloatForbidden { path: "$.a".into() },
            InspectorError::TokenAccountMismatch {
                computed: 1,
                provided: 2,
            },
            InspectorError::TokenOverflow,
            InspectorError::PairingMismatch {
                snapshot_id: Uuid::nil(),
                observation_id: Uuid::nil(),
            },
            InspectorError::InconsistentField { field: "model" },
            InspectorError::InvalidInput("bad".into()),
            InspectorError::Io("wal gone".into()),
            InspectorError::WalCorrupt {
                seq: 7,
                reason: "broken".into(),
            },
        ];
        for e in errs {
            let msg = e.to_string();
            assert!(!msg.is_empty(), "{e:?} 的 Display 不应为空");
        }
    }
}
