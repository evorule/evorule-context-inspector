//! 核心数据模型 —— 请求侧 / 响应侧分离（设计稿 §2，v2 修正版原样落地）。
//!
//! - `ContextSnapshot`：**请求侧**，发往 provider 前定格；
//! - `ResponseObservation`：**响应侧**，回包后追加，经 `response_to` 配对；
//! - `ContextSkeleton`：结构骨架，`skeleton_hash` 的唯一输入 ——
//!   只含结构 + 各内容的哈希，不含任何原文，默认模式下链验证即可从存储重算（强语义）。
//!
//! 确定性纪律（§2.7）：token 全 u64、计数 u32、时间 epoch 毫秒 u64、
//! 比率不落链（读取层派生）；骨架 tools 排序键 `(name, schema_hash)` 全序确定
//! （同名工具按 schema 哈希二次排序，输入顺序不影响哈希）。

use crate::canonical;
use crate::{
    BindingKind, ContentKind, FactId, HashHex, InspectorError, ProviderKind, SessionId, TokenSource,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 当前快照/观测/骨架的 schema 版本（设计稿 v2 时代）。
pub const SCHEMA_VERSION: u32 = 2;

/// 单条内容的哈希原语入口（B 族唯一合法入口 `evorule_hash::digest`）。
pub fn hash_content(content: &[u8]) -> HashHex {
    evorule_hash::digest(content)
}

/// 全量 messages+tools 原文的哈希（`--store-content` 时随快照记录，§2.5）。
///
/// 注意：`content_hash` **不是**链验证锚点（强验证只依赖 skeleton 重算），
/// 因此不走浮点拒收的 canonical 通道 —— provider 原文可含 `temperature: 0.7`
/// 之类合法浮点；此处哈希原始值本身，serde_json 的浮点最短往返表示实践稳定。
/// （哈希域登记见 evorule-hash README 注册表「上下文快照链」行。）
pub fn full_content_hash(messages_and_tools: &serde_json::Value) -> HashHex {
    evorule_hash::json_digest(messages_and_tools)
}

/// token 计数（全部 u64；压力等比率在读取层派生，不落链 — §2.7）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Composition {
    pub system_tokens: u64,
    pub tools_tokens: u64,
    pub messages_tokens: u64,
    pub images_tokens: u64,
    /// = 四分项之和；`with_total` 构造时核账，不等则拒收（§2.2）。
    pub total_tokens: u64,
    pub message_count: u32,
    pub tool_count: u32,
    pub token_source: TokenSource,
}

impl Composition {
    /// 分项构造：total 由分项 checked_add 派生（溢出报错）。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        system_tokens: u64,
        tools_tokens: u64,
        messages_tokens: u64,
        images_tokens: u64,
        message_count: u32,
        tool_count: u32,
        token_source: TokenSource,
    ) -> Result<Self, InspectorError> {
        let total = system_tokens
            .checked_add(tools_tokens)
            .and_then(|v| v.checked_add(messages_tokens))
            .and_then(|v| v.checked_add(images_tokens))
            .ok_or(InspectorError::TokenOverflow)?;
        Ok(Self {
            system_tokens,
            tools_tokens,
            messages_tokens,
            images_tokens,
            total_tokens: total,
            message_count,
            tool_count,
            token_source,
        })
    }

    /// 带显式 total 构造（外部计账来源）：核账不等即拒收（§2.2「不等则拒收」）。
    #[allow(clippy::too_many_arguments)]
    pub fn with_total(
        system_tokens: u64,
        tools_tokens: u64,
        messages_tokens: u64,
        images_tokens: u64,
        total_tokens: u64,
        message_count: u32,
        tool_count: u32,
        token_source: TokenSource,
    ) -> Result<Self, InspectorError> {
        let computed = Self::new(
            system_tokens,
            tools_tokens,
            messages_tokens,
            images_tokens,
            message_count,
            tool_count,
            token_source,
        )?
        .total_tokens;
        if computed != total_tokens {
            return Err(InspectorError::TokenAccountMismatch {
                computed,
                provided: total_tokens,
            });
        }
        Ok(Self {
            system_tokens,
            tools_tokens,
            messages_tokens,
            images_tokens,
            total_tokens,
            message_count,
            tool_count,
            token_source,
        })
    }
}

/// provider 回报的真实计数（§2.3）。字段全 Option：provider 未回报则 None。
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct UsageInfo {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// 支持 prompt cache 的 provider 才有（DeepSeek/Anthropic/OpenAI）。
    pub cached_tokens: Option<u64>,
}

/// 会话关联（§4.2 三级机制；Phase 3 对账的前提）。
///
/// **序列化口径**：`cause_fact` 以裸 JSON 数字落盘 —— 与主链
/// `fact_to_stable_json` 的 `"id": <u64>` 形态一致，跨链对账不引入第二种写法；
/// 因此 `SessionLink` 手写 serde（reactor 的 `FactId` 无 serde 派生）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionLink {
    /// 归属会话；未识别流量为 None（进 unattributed 桶，§4.2）。
    pub session_id: Option<SessionId>,
    /// 关联途径。
    pub binding: BindingKind,
    /// 可选：触发本次 LLM 调用的 evorule fact（审计对账用，反向引用）。
    pub cause_fact: Option<FactId>,
}

impl SessionLink {
    pub fn new(
        session_id: Option<SessionId>,
        binding: BindingKind,
        cause_fact: Option<FactId>,
    ) -> Self {
        Self {
            session_id,
            binding,
            cause_fact,
        }
    }
}

impl Serialize for SessionLink {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("SessionLink", 3)?;
        s.serialize_field("session_id", &self.session_id)?;
        s.serialize_field("binding", &self.binding)?;
        s.serialize_field("cause_fact", &self.cause_fact.map(|f| f.0))?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for SessionLink {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Wire {
            session_id: Option<SessionId>,
            binding: BindingKind,
            /// 主链口径：裸数字（fact_to_stable_json 的 id 形态）。
            cause_fact: Option<u64>,
        }
        let w = Wire::deserialize(deserializer)?;
        Ok(Self {
            session_id: w.session_id,
            binding: w.binding,
            cause_fact: w.cause_fact.map(FactId),
        })
    }
}

/// 骨架内单个工具条目（原文永不入模型 — 只入 schema 哈希）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolEntry {
    pub name: String,
    pub schema_hash: HashHex,
}

impl ToolEntry {
    pub fn new(name: impl Into<String>, schema: &[u8]) -> Self {
        Self {
            name: name.into(),
            schema_hash: hash_content(schema),
        }
    }
}

/// 骨架内单条消息条目。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageEntry {
    /// system / user / assistant / tool。
    pub role: String,
    pub content_kind: ContentKind,
    /// BLAKE3(该消息规范化原文)。
    pub content_hash: HashHex,
    /// 图片/附件数。
    pub media_count: u32,
}

impl MessageEntry {
    pub fn new(
        role: impl Into<String>,
        content_kind: ContentKind,
        content: &[u8],
        media_count: u32,
    ) -> Self {
        Self {
            role: role.into(),
            content_kind,
            content_hash: hash_content(content),
            media_count,
        }
    }
}

/// 快照的骨架：进入 `skeleton_hash` 的全部内容（§2.4）。
/// 原则：只含结构 + 各内容的哈希，不含任何原文 → 默认模式下链验证依然完备。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextSkeleton {
    pub schema_version: u32,
    /// `ProviderKind::as_str` 的稳定字符串。
    pub provider: String,
    pub model: String,
    /// BLAKE3(system prompt 原文) — 仅哈希。
    pub system_hash: HashHex,
    /// 按 `(name, schema_hash)` 字典序排序（确定性；计数信息保留 = multiset 语义）。
    pub tools: Vec<ToolEntry>,
    /// 保持原始顺序（顺序即信息）。
    pub messages: Vec<MessageEntry>,
}

impl ContextSkeleton {
    /// 构造即排序（tools）；`system_hash` 由原文哈希派生，原文不保存。
    pub fn new(
        provider: crate::ProviderKind,
        model: impl Into<String>,
        system_content: &[u8],
        mut tools: Vec<ToolEntry>,
        messages: Vec<MessageEntry>,
    ) -> Self {
        tools.sort_unstable_by(|a, b| (&a.name, &a.schema_hash).cmp(&(&b.name, &b.schema_hash)));
        Self {
            schema_version: SCHEMA_VERSION,
            provider: provider.as_str().to_owned(),
            model: model.into(),
            system_hash: hash_content(system_content),
            tools,
            messages,
        }
    }

    /// skeleton_hash = BLAKE3(canonical_json(self))（§2.4；哈希域「上下文快照链」）。
    /// 经 `canonical` 通道强制整数纪律；与 `evorule_hash::json_digest(self)`
    /// 字节等价（无浮点时两路同输入同输出，守护测试钉死）。
    pub fn skeleton_hash(&self) -> Result<HashHex, InspectorError> {
        let s = canonical::canonical_json(self)?;
        Ok(evorule_hash::digest(s.as_bytes()))
    }
}

/// 一次 LLM 调用发出前的上下文快照（请求侧，§2.2；只含结构，不含原文）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextSnapshot {
    pub schema_version: u32,
    /// 快照自身 ID（非 fact id）。
    pub snapshot_id: Uuid,
    pub session: SessionLink,
    /// epoch 毫秒（整数，保证确定性）。
    pub taken_at_epoch_ms: u64,
    pub provider: ProviderKind,
    pub model: String,
    /// 模型上限（tokens）；未知（如 Custom 未声明）为 None → pressure 置 None（§2.6）。
    pub context_window: Option<u32>,
    pub composition: Composition,
    pub skeleton: ContextSkeleton,
    /// BLAKE3(canonical_json(skeleton))，链验证锚点。
    pub skeleton_hash: HashHex,
    /// `--store-content` 开启时：全量 messages+tools 的 BLAKE3（原文旁挂，不入链）。
    pub content_hash: Option<HashHex>,
}

impl ContextSnapshot {
    /// 构造快照：`skeleton_hash` 在此定格；`snapshot.model` 必须与
    /// `skeleton.model` 一致（fail-fast，不静默收下矛盾数据）。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        snapshot_id: Uuid,
        session: SessionLink,
        taken_at_epoch_ms: u64,
        provider: ProviderKind,
        model: impl Into<String>,
        context_window: Option<u32>,
        composition: Composition,
        skeleton: ContextSkeleton,
    ) -> Result<Self, InspectorError> {
        let model = model.into();
        if skeleton.model != model {
            return Err(InspectorError::InconsistentField {
                field: "model(snapshot vs skeleton)",
            });
        }
        let skeleton_hash = skeleton.skeleton_hash()?;
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            snapshot_id,
            session,
            taken_at_epoch_ms,
            provider,
            model,
            context_window,
            composition,
            skeleton,
            skeleton_hash,
            content_hash: None,
        })
    }

    /// 响应回填（§2.3）：配对校验后，`usage.input_tokens` 有值则将
    /// `composition.token_source` 升级为 `ProviderReported`（估算分项保留供对照）。
    /// 返回是否发生升级（幂等：已是 ProviderReported 或无 input 回报 → false）。
    pub fn backfill_token_source(
        &mut self,
        observation: &ResponseObservation,
    ) -> Result<bool, InspectorError> {
        if observation.response_to != self.snapshot_id {
            return Err(InspectorError::PairingMismatch {
                snapshot_id: self.snapshot_id,
                observation_id: observation.observation_id,
            });
        }
        if observation.usage.input_tokens.is_some()
            && self.composition.token_source != TokenSource::ProviderReported
        {
            self.composition.token_source = TokenSource::ProviderReported;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// 一次 LLM 调用的响应观测（响应侧，§2.3；流式请求在末 chunk 汇总，§4.3）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseObservation {
    pub schema_version: u32,
    pub observation_id: Uuid,
    /// 配对的 snapshot_id。
    pub response_to: Uuid,
    pub received_at_epoch_ms: u64,
    /// "length" / "max_tokens" / "stop" / ...
    pub finish_reason: Option<String>,
    pub usage: UsageInfo,
    /// 截断检测（§3.4：finish_reason 路径 + 可配正则）。
    pub truncation_detected: bool,
    /// 是否流式响应。
    pub stream: bool,
}

impl ResponseObservation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        observation_id: Uuid,
        response_to: Uuid,
        received_at_epoch_ms: u64,
        finish_reason: Option<String>,
        usage: UsageInfo,
        truncation_detected: bool,
        stream: bool,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            observation_id,
            response_to,
            received_at_epoch_ms,
            finish_reason,
            usage,
            truncation_detected,
            stream,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn skeleton_fixture() -> ContextSkeleton {
        let tools = vec![
            ToolEntry::new("weather", br#"{"name":"weather"}"#),
            ToolEntry::new("calc", br#"{"name":"calc"}"#),
        ];
        let messages = vec![
            MessageEntry::new("user", ContentKind::Text, b"hello world", 0),
            MessageEntry::new("assistant", ContentKind::ToolCall, b"call weather", 0),
        ];
        ContextSkeleton::new(
            ProviderKind::DeepSeek,
            "deepseek-chat",
            b"You are a helpful assistant.",
            tools,
            messages,
        )
    }

    fn composition_fixture() -> Composition {
        Composition::new(120, 300, 900, 0, 2, 2, TokenSource::Estimated).unwrap()
    }

    // ---- Composition 核账（验收：总 token 核账拒收）----

    #[test]
    fn composition_derives_total_from_parts() {
        let c = composition_fixture();
        assert_eq!(c.total_tokens, 1320);
    }

    #[test]
    fn composition_with_total_rejects_mismatch() {
        let err = Composition::with_total(120, 300, 900, 0, 999, 2, 2, TokenSource::Estimated)
            .unwrap_err();
        assert_eq!(
            err,
            InspectorError::TokenAccountMismatch {
                computed: 1320,
                provided: 999
            }
        );
    }

    #[test]
    fn composition_with_total_accepts_exact_sum() {
        let c =
            Composition::with_total(120, 300, 900, 0, 1320, 2, 2, TokenSource::Estimated).unwrap();
        assert_eq!(c.total_tokens, 1320);
    }

    #[test]
    fn composition_rejects_u64_overflow() {
        let big = u64::MAX;
        assert_eq!(
            Composition::new(big, 1, 0, 0, 0, 0, TokenSource::Estimated).unwrap_err(),
            InspectorError::TokenOverflow
        );
    }

    #[test]
    fn composition_with_total_propagates_overflow() {
        assert_eq!(
            Composition::with_total(u64::MAX, 1, 0, 0, 0, 0, 0, TokenSource::Estimated)
                .unwrap_err(),
            InspectorError::TokenOverflow
        );
    }

    #[test]
    fn full_content_hash_hashes_raw_value_float_included() {
        // content_hash 面对原文（可含合法浮点，如 temperature），哈希原始序列化值；
        // 与 B 族 json_digest 同一字节口径（域见 evorule-hash README 注册表）。
        let raw = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7
        });
        assert_eq!(full_content_hash(&raw), evorule_hash::json_digest(&raw));
        // 同输入确定性
        assert_eq!(full_content_hash(&raw), full_content_hash(&raw));
    }

    // ---- Skeleton 排序确定性与哈希（验收 #10 命门测试之本地半）----

    #[test]
    fn skeleton_tools_sorted_by_name_then_schema_hash() {
        let tools = vec![
            ToolEntry::new("z_tool", br#"{"z":1}"#),
            ToolEntry::new("a_tool", br#"{"a":2}"#),
            ToolEntry::new("a_tool", br#"{"a":1}"#),
        ];
        let s = ContextSkeleton::new(ProviderKind::OpenAi, "gpt-x", b"sys", tools, vec![]);
        let names: Vec<&str> = s.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["a_tool", "a_tool", "z_tool"]);
        // 同名 tie-break：schema_hash 字典序
        assert!(s.tools[0].schema_hash <= s.tools[1].schema_hash);
    }

    #[test]
    fn skeleton_hash_is_stable_under_reordering() {
        let s = skeleton_fixture();
        let h1 = s.skeleton_hash().unwrap();
        let h2 = ContextSkeleton::new(
            ProviderKind::DeepSeek,
            "deepseek-chat",
            b"You are a helpful assistant.",
            // 乱序重进 → 排序后同骨架
            vec![
                ToolEntry::new("weather", br#"{"name":"weather"}"#),
                ToolEntry::new("calc", br#"{"name":"calc"}"#),
            ],
            s.messages.clone(),
        )
        .skeleton_hash()
        .unwrap();
        assert_eq!(h1, h2, "同内容乱序输入必须同哈希（确定性红线）");
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn skeleton_hash_walks_canonical_channel_exactly() {
        // skeleton_hash 必须逐字节 = digest(canonical_json(skeleton))，
        // 不得旁路 canonical 通道（整数纪律执行点不可绕过）。
        let s = skeleton_fixture();
        let expected =
            evorule_hash::digest(crate::canonical::canonical_json(&s).unwrap().as_bytes());
        assert_eq!(s.skeleton_hash().unwrap(), expected);
        // 注意：这里**刻意不**断言与 `json_digest(&s)` 相等 ——
        // json_digest 直哈希派生 Serialize 结构体时按字段声明序输出，
        // 而 canonical 通道经 BTreeMap 输出字典序（设计稿 §2.5 对齐 reactor
        // fact_to_stable_json 的 Map 键序）。字典序才是本域 canonical 口径。
    }

    // ---- 黄金向量（验收 #10：Windows 本地钉值，CI Linux 复算比对）----

    #[test]
    fn golden_vector_canonical_skeleton_hash() {
        // 固定输入：跨平台唯一允许的差异来源是序列化；此处钉死字节与哈希。
        let s = ContextSkeleton::new(
            ProviderKind::Anthropic,
            "claude-sonnet",
            b"system prompt v1",
            vec![ToolEntry::new("search", br#"{"description":"search"}"#)],
            vec![
                MessageEntry::new("user", ContentKind::Text, b"question", 0),
                MessageEntry::new("assistant", ContentKind::Text, b"answer", 0),
            ],
        );
        let canonical = crate::canonical::canonical_json(&s).unwrap();
        assert_eq!(
            canonical,
            "{\"messages\":[{\"content_hash\":\"2bfb541601a90051abc4133e5f0bd45b4f2432b47000dc08bc8bcca10a57606e\",\"content_kind\":\"text\",\"media_count\":0,\"role\":\"user\"},{\"content_hash\":\"63b24a318813c47bce54f4e4dc1f052a586c63eeb96d20ed88bf1b86bd6d362c\",\"content_kind\":\"text\",\"media_count\":0,\"role\":\"assistant\"}],\"model\":\"claude-sonnet\",\"provider\":\"anthropic\",\"schema_version\":2,\"system_hash\":\"448aa4e01bbe7d9962cc1208fb8982f2d2efab53d43db4bba786df79d0d7a51b\",\"tools\":[{\"name\":\"search\",\"schema_hash\":\"e30e60aa86b644891e6fe7d7527b1126eb46d63305031ea441b868cd8c488ef9\"}]}"
        );
        let h = s.skeleton_hash().unwrap();
        assert_eq!(
            h,
            "0d4784d129c4b7734e0ef034e90ca1369d81101346e9c29c50ecc0b09110e004"
        );
    }

    // ---- Snapshot 构造与回填（验收：TokenSource 升级）----

    fn snapshot_fixture() -> ContextSnapshot {
        let link = SessionLink::new(Some(42), BindingKind::PortBinding, Some(FactId(123)));
        ContextSnapshot::new(
            Uuid::nil(),
            link,
            1_760_000_000_000,
            ProviderKind::DeepSeek,
            "deepseek-chat",
            Some(131_072),
            composition_fixture(),
            skeleton_fixture(),
        )
        .unwrap()
    }

    #[test]
    fn snapshot_new_computes_skeleton_hash() {
        let snap = snapshot_fixture();
        assert_eq!(snap.skeleton_hash, snap.skeleton.skeleton_hash().unwrap());
        assert_eq!(snap.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn snapshot_new_rejects_model_mismatch() {
        let skeleton =
            ContextSkeleton::new(ProviderKind::DeepSeek, "other-model", b"s", vec![], vec![]);
        let err = ContextSnapshot::new(
            Uuid::nil(),
            SessionLink::new(None, BindingKind::Unattributed, None),
            0,
            ProviderKind::DeepSeek,
            "deepseek-chat",
            None,
            composition_fixture(),
            skeleton,
        )
        .unwrap_err();
        assert!(matches!(err, InspectorError::InconsistentField { .. }));
    }

    #[test]
    fn backfill_upgrades_token_source() {
        let mut snap = snapshot_fixture();
        assert_eq!(snap.composition.token_source, TokenSource::Estimated);
        let obs = ResponseObservation::new(
            Uuid::nil(),
            snap.snapshot_id,
            1_760_000_001_000,
            Some("stop".into()),
            UsageInfo {
                input_tokens: Some(1290),
                output_tokens: Some(88),
                cached_tokens: None,
            },
            false,
            false,
        );
        assert!(snap.backfill_token_source(&obs).unwrap());
        assert_eq!(snap.composition.token_source, TokenSource::ProviderReported);
        // 幂等
        assert!(!snap.backfill_token_source(&obs).unwrap());
    }

    #[test]
    fn backfill_without_input_report_leaves_estimated() {
        let mut snap = snapshot_fixture();
        let obs = ResponseObservation::new(
            Uuid::nil(),
            snap.snapshot_id,
            0,
            Some("stop".into()),
            UsageInfo {
                input_tokens: None,
                output_tokens: Some(1),
                cached_tokens: None,
            },
            false,
            false,
        );
        assert!(!snap.backfill_token_source(&obs).unwrap());
        assert_eq!(snap.composition.token_source, TokenSource::Estimated);
    }

    #[test]
    fn backfill_rejects_pairing_mismatch() {
        let mut snap = snapshot_fixture();
        let obs = ResponseObservation::new(
            Uuid::nil(),
            Uuid::new_v4(),
            0,
            None,
            UsageInfo {
                input_tokens: Some(1),
                output_tokens: None,
                cached_tokens: None,
            },
            false,
            false,
        );
        assert!(matches!(
            snap.backfill_token_source(&obs).unwrap_err(),
            InspectorError::PairingMismatch { .. }
        ));
        assert_eq!(snap.composition.token_source, TokenSource::Estimated);
    }

    // ---- 序列化口径（SessionLink cause_fact = 主链裸数字形态）----

    #[test]
    fn session_link_serializes_cause_fact_as_bare_number() {
        let link = SessionLink::new(Some(42), BindingKind::Header, Some(FactId(123)));
        let s = serde_json::to_string(&link).unwrap();
        assert_eq!(
            s,
            r#"{"session_id":42,"binding":"header","cause_fact":123}"#
        );
        let back: SessionLink = serde_json::from_str(&s).unwrap();
        assert_eq!(back, link);
    }

    #[test]
    fn snapshot_roundtrips_through_serde() {
        let snap = snapshot_fixture();
        let s = serde_json::to_string(&snap).unwrap();
        let back: ContextSnapshot = serde_json::from_str(&s).unwrap();
        assert_eq!(back, snap);
    }
}
