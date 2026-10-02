//! LLM Provider 抽象（设计稿 §4）—— 从 LLM API 请求/响应体抽取结构。
//!
//! trait 全同步（纯 JSON 解析，无 I/O）；HTTP 层与 SSE 转发属 Phase 3 插件进程。
//! `StreamAggregator` 把逐 chunk 的 `StreamEvent` 汇总为响应侧观测的
//! usage / finish_reason —— usage 按**字段合并**（Anthropic 的 input 在
//! message_start、output 在 message_delta，两段拼合而非覆盖）。

pub mod anthropic;
pub mod deepseek;
pub mod ollama;
pub mod openai;

use crate::model::{ToolEntry, UsageInfo};
use crate::{ContentKind, InspectorError, ProviderKind};
use serde_json::Value;

/// 工具 schema（原文内存，哈希在骨架阶段）。
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSchema {
    pub name: String,
    /// 工具定义对象（provider 原生形态；骨架取其 canonical 字节哈希）。
    pub definition: Value,
}

/// 单条抽取后的消息（原文内存，永不落链 — 只入哈希与旁挂存储）。
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedMessage {
    pub role: String,
    pub content_kind: ContentKind,
    /// 该消息规范化文本（多 part 时按序拼接）。
    pub text: String,
    /// 图片/附件数。
    pub media_count: u32,
}

/// 请求体抽取结果（`stream_options` 注入等改写**之后**的 `raw` 才是实发请求）。
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedRequest {
    pub model: String,
    /// system prompt 原文（多条按序 `\n` 拼接；无则 None）。
    pub system: Option<String>,
    pub tools: Vec<ToolSchema>,
    pub messages: Vec<ExtractedMessage>,
    /// provider 请求选项（如 Ollama `options.num_ctx`）。
    pub request_options: Option<Value>,
    /// 实发完整请求体（改写后），骨架与全文哈希的唯一输入。
    pub raw: Value,
}

/// 图片引用（多模态计费口径输入）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageRef {
    pub width: u32,
    pub height: u32,
    pub high_detail: bool,
}

/// token 估算分项（total 由 `Composition::new` 派生核账）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenCounts {
    pub system_tokens: u64,
    pub tools_tokens: u64,
    pub messages_tokens: u64,
    pub images_tokens: u64,
}

/// 非流式响应抽取结果（usage + finish_reason；截断判定由 metrics 层依据二者）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResponseExtraction {
    pub usage: UsageInfo,
    pub finish_reason: Option<String>,
}

/// 流式单 chunk 事件分类（设计稿 §4 + v3.2 增补 `Final`）。
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// 中间内容 chunk（观测但不落账）。
    ContentDelta,
    /// usage 携带 chunk —— **字段级合并**语义：只覆盖出现的字段
    /// （Anthropic input 在 message_start、output 在 message_delta）。
    UsageFinal(UsageInfo),
    /// 结束原因（"length" / "stop" / "max_tokens" / "end_turn" / ...）。
    FinishReason(String),
    /// 末 chunk **同时**携带 usage 与 finish_reason（Anthropic `message_delta`
    /// 单 chunk 双字段 —— 设计稿原枚举表达不了这一事实，v3.2 增补本变体）。
    Final {
        usage: Option<UsageInfo>,
        finish_reason: Option<String>,
    },
    /// 未知 chunk：容错记录，不炸（fail-soft 仅限观测面，见 §9 SSE 容错）。
    Unrecognized,
}

/// 流式聚合器：逐 chunk 观察，收尾产出 usage + finish_reason（§4.3）。
#[derive(Debug, Default)]
pub struct StreamAggregator {
    usage: UsageInfo,
    finish_reason: Option<String>,
    content_chunks: u64,
}

/// 聚合结果（配对成 `ResponseObservation` 由调用方完成，stream=true）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AggregatedStream {
    pub usage: UsageInfo,
    pub finish_reason: Option<String>,
    pub content_chunks: u64,
}

impl StreamAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::ContentDelta => self.content_chunks += 1,
            StreamEvent::UsageFinal(u) => self.merge_usage(u),
            StreamEvent::Final {
                usage,
                finish_reason,
            } => {
                if let Some(u) = usage {
                    self.merge_usage(u);
                }
                if let Some(r) = finish_reason {
                    self.finish_reason = Some(r);
                }
            }
            StreamEvent::FinishReason(r) => self.finish_reason = Some(r),
            StreamEvent::Unrecognized => {}
        }
    }

    fn merge_usage(&mut self, u: UsageInfo) {
        if u.input_tokens.is_some() {
            self.usage.input_tokens = u.input_tokens;
        }
        if u.output_tokens.is_some() {
            self.usage.output_tokens = u.output_tokens;
        }
        if u.cached_tokens.is_some() {
            self.usage.cached_tokens = u.cached_tokens;
        }
    }

    pub fn finish(self) -> AggregatedStream {
        AggregatedStream {
            usage: self.usage,
            finish_reason: self.finish_reason,
            content_chunks: self.content_chunks,
        }
    }
}

/// LLM Provider 抽象（同步 trait；插件进程内按协议分发）。
pub trait LlmProvider: Send + Sync {
    fn kind(&self) -> ProviderKind;

    /// 从 LLM API 请求体抽取 messages + tools（→ ContextSkeleton + Composition 的原料）。
    fn extract_request(&self, request_body: &Value) -> Result<ExtractedRequest, InspectorError>;

    /// 从非流式响应体抽取 usage + finish_reason。
    fn extract_response(&self, response_body: &Value)
        -> Result<ResponseExtraction, InspectorError>;

    /// 流式响应逐 chunk 事件抽取（SSE data 行 / Ollama NDJSON 行）。
    fn extract_stream_event(&self, chunk: &Value) -> StreamEvent;

    /// token 估算（绑定 tiktoken，不自研分词）。
    fn count_tokens(&self, req: &ExtractedRequest) -> Result<TokenCounts, InspectorError>;

    /// 图片 token（多模态，provider 各有计法）。
    fn count_image_tokens(&self, image: &ImageRef) -> u64;

    /// context_window 解析（§2.6）：常量表兜底；Ollama 读 num_ctx；
    /// 未知模型 → None（pressure 置 None）。
    fn context_window(&self, model: &str, request_options: Option<&Value>) -> Option<u32>;
}

/// 工具 schema 的 canonical 字节（哈希输入；字典序 + 浮点拒收口径）。
pub fn tool_schema_bytes(definition: &Value) -> Result<Vec<u8>, InspectorError> {
    Ok(crate::canonical::canonical_json(definition)?.into_bytes())
}

/// 从 `ExtractedRequest` 构建骨架 tools 条目（排序由骨架构造器完成）。
pub fn skeleton_tools(req: &ExtractedRequest) -> Result<Vec<ToolEntry>, InspectorError> {
    req.tools
        .iter()
        .map(|t| {
            Ok(ToolEntry::new(
                t.name.clone(),
                &tool_schema_bytes(&t.definition)?,
            ))
        })
        .collect()
}

/// 按 provider 种类静态分发 `context_window`（无 provider 实例在手时的便捷入口；
/// Custom 未声明 → None，pressure 置 None）。
pub fn window_of(kind: ProviderKind, model: &str, options: Option<&Value>) -> Option<u32> {
    match kind {
        ProviderKind::OpenAi => openai::OpenAiProvider.context_window(model, options),
        ProviderKind::Anthropic => anthropic::AnthropicProvider.context_window(model, options),
        ProviderKind::DeepSeek => deepseek::DeepSeekProvider.context_window(model, options),
        ProviderKind::Ollama => ollama::OllamaProvider.context_window(model, options),
        ProviderKind::Custom => None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use serde_json::json;

    #[test]
    fn aggregator_merges_usage_fields_not_replaces() {
        let mut agg = StreamAggregator::new();
        // Anthropic 形态：message_start 只带 input + cache，message_delta 只带 output
        agg.observe(StreamEvent::UsageFinal(UsageInfo {
            input_tokens: Some(120),
            output_tokens: None,
            cached_tokens: Some(64),
        }));
        agg.observe(StreamEvent::ContentDelta);
        agg.observe(StreamEvent::ContentDelta);
        agg.observe(StreamEvent::UsageFinal(UsageInfo {
            input_tokens: None,
            output_tokens: Some(88),
            cached_tokens: None,
        }));
        agg.observe(StreamEvent::FinishReason("end_turn".into()));
        let out = agg.finish();
        assert_eq!(out.usage.input_tokens, Some(120));
        assert_eq!(out.usage.output_tokens, Some(88));
        assert_eq!(out.usage.cached_tokens, Some(64));
        assert_eq!(out.finish_reason.as_deref(), Some("end_turn"));
        assert_eq!(out.content_chunks, 2);
    }

    #[test]
    fn aggregator_empty_is_default() {
        let out = StreamAggregator::new().finish();
        assert_eq!(out, AggregatedStream::default());
    }

    #[test]
    fn skeleton_tools_ok_path_hashes_definitions() {
        let req = ExtractedRequest {
            model: "m".into(),
            system: None,
            tools: vec![ToolSchema {
                name: "t".into(),
                definition: json!({"type": "object"}),
            }],
            messages: vec![],
            request_options: None,
            raw: json!({}),
        };
        let tools = skeleton_tools(&req).unwrap_or_else(|e| panic!("skeleton_tools 失败: {e}"));
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "t");
        assert_eq!(tools[0].schema_hash.len(), 64);
    }

    #[test]
    fn aggregator_final_variants() {
        let mut agg = StreamAggregator::new();
        agg.observe(StreamEvent::Final {
            usage: None,
            finish_reason: Some("stop".into()),
        });
        assert_eq!(agg.finish().finish_reason.as_deref(), Some("stop"));
        let mut agg2 = StreamAggregator::new();
        agg2.observe(StreamEvent::Final {
            usage: Some(UsageInfo {
                input_tokens: Some(1),
                output_tokens: None,
                cached_tokens: None,
            }),
            finish_reason: None,
        });
        let out = agg2.finish();
        assert_eq!(out.usage.input_tokens, Some(1));
        assert_eq!(out.finish_reason, None);
    }

    #[test]
    fn tool_schema_bytes_rejects_float_definitions() {
        let err = tool_schema_bytes(&json!({"a": 1.5}));
        assert!(err.is_err(), "工具 schema 含浮点必须被 canonical 通道拒收");
        let ok = tool_schema_bytes(&json!({"a": 1}));
        assert!(ok.is_ok());
    }

    #[test]
    fn aggregator_unrecognized_is_ignored_but_counted_neither() {
        let mut agg = StreamAggregator::new();
        agg.observe(StreamEvent::Unrecognized);
        let out = agg.finish();
        assert_eq!(out.content_chunks, 0);
        assert_eq!(out.usage, UsageInfo::default());
    }
}

#[cfg(test)]
mod window_of_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use serde_json::json;

    #[test]
    fn window_of_dispatches_all_kinds() {
        let opts = json!({"num_ctx": 4096});
        assert_eq!(
            window_of(ProviderKind::OpenAi, "gpt-4o", None),
            Some(128_000)
        );
        assert_eq!(
            window_of(ProviderKind::Anthropic, "claude-sonnet-4", None),
            Some(200_000)
        );
        assert_eq!(
            window_of(ProviderKind::DeepSeek, "deepseek-chat", None),
            Some(128_000)
        );
        assert_eq!(
            window_of(ProviderKind::Ollama, "llama3.2", Some(&opts)),
            Some(4096)
        );
        assert_eq!(window_of(ProviderKind::Custom, "anything", None), None);
    }
}
