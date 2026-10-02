//! OpenAI 协议实现（含 DeepSeek 协议兼容共用路径；SSE 同路径）。
//!
//! 请求体形态：`{model, messages:[{role, content}], tools?:[{type:"function",
//! function:{name,description,parameters}}], stream_options?:{include_usage}}`。
//! 响应：`{choices:[{finish_reason, message}], usage:{prompt_tokens,
//! completion_tokens, prompt_tokens_details?{cached_tokens}}}`。
//! SSE：delta chunk（finish_reason=null）→ finish chunk → 空 choices 的
//! usage chunk（include_usage）→ `data: [DONE]`。

use super::{
    ExtractedMessage, ExtractedRequest, ImageRef, LlmProvider, ResponseExtraction, StreamEvent,
    TokenCounts, ToolSchema,
};
use crate::model::UsageInfo;
use crate::tokens;
use crate::{ContentKind, InspectorError, ProviderKind};
use serde_json::Value;

/// 已知模型窗口常量表（tokens；保守口径，动态查询为 Phase 3 增强）。
fn builtin_context_window(model: &str) -> Option<u32> {
    let m = model.to_ascii_lowercase();
    // 前缀匹配容忍 -20241120 之类日期后缀
    if m.starts_with("gpt-4o") || m.starts_with("gpt-4.1") || m.starts_with("chatgpt-4o") {
        Some(128_000)
    } else if m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4") {
        Some(200_000)
    } else if m.starts_with("gpt-4-turbo") || m.starts_with("gpt-4-") {
        Some(128_000)
    } else if m.starts_with("gpt-4") {
        Some(8_192)
    } else if m.starts_with("gpt-3.5-turbo") {
        Some(16_385)
    } else {
        None
    }
}

/// 从 OpenAI 形态 messages 数组抽取（system/developer 归 system，其余保序）。
fn extract_messages(
    messages: &Value,
) -> Result<(Option<String>, Vec<ExtractedMessage>), InspectorError> {
    let arr = messages
        .as_array()
        .ok_or_else(|| InspectorError::InvalidInput("messages 不是数组".into()))?;
    let mut system_parts: Vec<String> = Vec::new();
    let mut out: Vec<ExtractedMessage> = Vec::new();
    for (i, m) in arr.iter().enumerate() {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| InspectorError::InvalidInput(format!("messages[{i}] 缺 role")))?
            .to_owned();
        let (text, mut kind, media) = parse_content(m.get("content").unwrap_or(&Value::Null))?;
        // kind 语义（§2.4）：tool role = 工具结果；带 tool_calls 的 assistant = 工具调用
        if role == "tool" {
            kind = ContentKind::ToolResult;
        } else if m.get("tool_calls").is_some() {
            kind = ContentKind::ToolCall;
        }
        if role == "system" || role == "developer" {
            system_parts.push(text);
        } else {
            out.push(ExtractedMessage {
                role,
                content_kind: kind,
                text,
                media_count: media,
            });
        }
    }
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n"))
    };
    Ok((system, out))
}

/// content 字段：字符串 或 多 part 数组（text / image_url）。
fn parse_content(content: &Value) -> Result<(String, ContentKind, u32), InspectorError> {
    match content {
        Value::String(s) => Ok((s.clone(), ContentKind::Text, 0)),
        Value::Array(parts) => {
            let mut text = String::new();
            let mut media = 0u32;
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(t);
                        }
                    }
                    Some("image_url") | Some("input_image") => media += 1,
                    _ => {}
                }
            }
            let kind = if media > 0 && !text.is_empty() {
                ContentKind::Mixed
            } else if media > 0 {
                ContentKind::Image
            } else {
                ContentKind::Text
            };
            Ok((text, kind, media))
        }
        Value::Null => Ok((String::new(), ContentKind::Text, 0)),
        _ => Err(InspectorError::InvalidInput("content 形态不支持".into())),
    }
}

/// OpenAI 协议 provider（DeepSeek 兼容路径复用本实现的抽取逻辑）。
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAiProvider;

impl OpenAiProvider {
    fn extract_request_impl(&self, body: &Value) -> Result<ExtractedRequest, InspectorError> {
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| InspectorError::InvalidInput("缺 model".into()))?
            .to_owned();
        let messages_value = body
            .get("messages")
            .ok_or_else(|| InspectorError::InvalidInput("缺 messages".into()))?;
        let (system, messages) = extract_messages(messages_value)?;
        let mut tools = Vec::new();
        if let Some(arr) = body.get("tools").and_then(Value::as_array) {
            for t in arr {
                // OpenAI: {type:"function", function:{name, description, parameters}}
                let def = t.get("function").unwrap_or(t); // 兼容直接平铺 function 对象
                let name = def
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| InspectorError::InvalidInput("tools[] 缺 name".into()))?
                    .to_owned();
                tools.push(ToolSchema {
                    name,
                    definition: def.clone(),
                });
            }
        }
        Ok(ExtractedRequest {
            model,
            system,
            tools,
            messages,
            request_options: None,
            raw: body.clone(),
        })
    }

    fn extract_response_impl(&self, body: &Value) -> Result<ResponseExtraction, InspectorError> {
        let choice = body
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
            .ok_or_else(|| InspectorError::InvalidInput("缺 choices[0]".into()))?;
        let finish_reason = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let usage = body.get("usage").map(parse_usage).unwrap_or_default();
        Ok(ResponseExtraction {
            usage,
            finish_reason,
        })
    }

    fn extract_stream_event_impl(&self, chunk: &Value) -> StreamEvent {
        // 实测协议差异（真实 fixture openai_stream.sse，MiniMax 2026-10-02）：
        // 标准 OpenAI 末 chunk = 空 choices + usage；MiniMax 末 chunk = delta+
        // finish_reason+usage 同 chunk，且中间 chunk 也带全空 usage 字段
        // （解析后全 None，字段合并为无害 no-op）。统一按组合语义处理。
        let choice = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first());
        let finish_reason = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let usage = chunk.get("usage").filter(|u| !u.is_null()).map(parse_usage);
        match (usage, finish_reason) {
            (Some(u), Some(r)) => StreamEvent::Final {
                usage: Some(u),
                finish_reason: Some(r),
            },
            (Some(u), None) => {
                if choice.is_none() {
                    StreamEvent::UsageFinal(u)
                } else if u.input_tokens.is_none()
                    && u.output_tokens.is_none()
                    && u.cached_tokens.is_none()
                {
                    // MiniMax 中间 chunk：delta + 全空 usage 字段 → 内容 chunk
                    StreamEvent::ContentDelta
                } else {
                    StreamEvent::Final {
                        usage: Some(u),
                        finish_reason: None,
                    }
                }
            }
            (None, Some(r)) => StreamEvent::FinishReason(r),
            (None, None) => {
                if choice.and_then(|c| c.get("delta")).is_some() {
                    StreamEvent::ContentDelta
                } else {
                    StreamEvent::Unrecognized
                }
            }
        }
    }

    fn count_tokens_impl(&self, req: &ExtractedRequest) -> Result<TokenCounts, InspectorError> {
        count_with_tiktoken(ProviderKind::OpenAi, req)
    }
}

impl LlmProvider for OpenAiProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAi
    }

    fn extract_request(&self, request_body: &Value) -> Result<ExtractedRequest, InspectorError> {
        self.extract_request_impl(request_body)
    }

    fn extract_response(
        &self,
        response_body: &Value,
    ) -> Result<ResponseExtraction, InspectorError> {
        self.extract_response_impl(response_body)
    }

    fn extract_stream_event(&self, chunk: &Value) -> StreamEvent {
        self.extract_stream_event_impl(chunk)
    }

    fn count_tokens(&self, req: &ExtractedRequest) -> Result<TokenCounts, InspectorError> {
        self.count_tokens_impl(req)
    }

    fn count_image_tokens(&self, image: &ImageRef) -> u64 {
        tokens::estimate_image_tokens(
            ProviderKind::OpenAi,
            image.width,
            image.height,
            image.high_detail,
        )
    }

    fn context_window(&self, model: &str, _request_options: Option<&Value>) -> Option<u32> {
        builtin_context_window(model)
    }
}

/// OpenAI usage 块解析（兼容 DeepSeek：多出的 cache 字段在 DeepSeekProvider 处理）。
pub(crate) fn parse_usage(u: &Value) -> UsageInfo {
    let cached = u
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64);
    UsageInfo {
        input_tokens: u.get("prompt_tokens").and_then(Value::as_u64),
        output_tokens: u.get("completion_tokens").and_then(Value::as_u64),
        cached_tokens: cached,
    }
}

pub(crate) fn count_with_tiktoken(
    provider_kind: ProviderKind,
    req: &ExtractedRequest,
) -> Result<TokenCounts, InspectorError> {
    let model = req.model.as_str();
    let system_tokens = match &req.system {
        Some(s) => tokens::estimate_text(model, s)?,
        None => 0,
    };
    let mut tools_tokens = 0u64;
    for t in &req.tools {
        let bytes = super::tool_schema_bytes(&t.definition)?;
        let s = std::str::from_utf8(&bytes).unwrap_or_default();
        tools_tokens += tokens::estimate_text(model, s)?;
    }
    let mut messages_tokens = 0u64;
    for m in &req.messages {
        messages_tokens += tokens::estimate_text(model, &m.text)?;
    }
    let images_tokens = 0u64;
    let _ = provider_kind; // 图片按调用方逐张计（count_image_tokens），此处不批量估
    Ok(TokenCounts {
        system_tokens,
        tools_tokens,
        messages_tokens,
        images_tokens,
    })
}

/// `stream_options.include_usage` 注入（透传前改写；改写后才算骨架 — §4.3）。
/// 幂等：已声明即不重复注入。返回是否发生修改。
pub fn inject_stream_options_include_usage(body: &mut Value) -> bool {
    let obj = match body.as_object_mut() {
        Some(o) => o,
        None => return false,
    };
    let already = obj
        .get("stream_options")
        .and_then(|so| so.get("include_usage"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if already {
        return false;
    }
    match obj.get_mut("stream_options") {
        Some(Value::Object(so)) => {
            so.insert("include_usage".into(), Value::Bool(true));
        }
        _ => {
            obj.insert(
                "stream_options".into(),
                serde_json::json!({"include_usage": true}),
            );
        }
    }
    true
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use serde_json::json;

    #[test]
    fn inject_is_idempotent_and_creates_when_missing() {
        let mut body = json!({"model": "gpt-4o", "messages": []});
        assert!(inject_stream_options_include_usage(&mut body));
        assert_eq!(body["stream_options"]["include_usage"], json!(true));
        // 已有 include_usage=false → 升级为 true
        let mut b2 = json!({"stream_options": {"include_usage": false}});
        assert!(inject_stream_options_include_usage(&mut b2));
        assert_eq!(b2["stream_options"]["include_usage"], json!(true));
        // 幂等
        assert!(!inject_stream_options_include_usage(&mut body));
    }

    #[test]
    fn context_window_prefix_tolerates_date_suffix() {
        assert_eq!(
            OpenAiProvider.context_window("gpt-4o-2024-11-20", None),
            Some(128_000)
        );
        assert_eq!(OpenAiProvider.context_window("mystery-model", None), None);
    }

    #[test]
    fn extract_request_system_and_tools() {
        let p = OpenAiProvider;
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]}
            ],
            "tools": [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object"}}}]
        });
        let req = p.extract_request(&body).unwrap();
        assert_eq!(req.system.as_deref(), Some("be terse"));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].content_kind, ContentKind::Mixed);
        assert_eq!(req.messages[0].media_count, 1);
        assert_eq!(req.tools[0].name, "calc");
    }

    #[test]
    fn extract_response_and_stream_usage_chunk() {
        let p = OpenAiProvider;
        let resp = json!({
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 28, "completion_tokens": 142, "prompt_tokens_details": {"cached_tokens": 0}}
        });
        let ex = p.extract_response(&resp).unwrap();
        assert_eq!(ex.finish_reason.as_deref(), Some("stop"));
        assert_eq!(ex.usage.input_tokens, Some(28));

        let usage_chunk =
            json!({"choices": [], "usage": {"prompt_tokens": 48, "completion_tokens": 90}});
        assert_eq!(
            p.extract_stream_event(&usage_chunk),
            StreamEvent::UsageFinal(UsageInfo {
                input_tokens: Some(48),
                output_tokens: Some(90),
                cached_tokens: None
            })
        );
        let delta =
            json!({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": null}]});
        assert_eq!(p.extract_stream_event(&delta), StreamEvent::ContentDelta);
        let done = json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]});
        assert_eq!(
            p.extract_stream_event(&done),
            StreamEvent::FinishReason("length".into())
        );
    }

    #[test]
    fn context_window_full_table() {
        let p = OpenAiProvider;
        assert_eq!(p.context_window("o1-mini", None), Some(200_000));
        assert_eq!(p.context_window("o3", None), Some(200_000));
        assert_eq!(p.context_window("gpt-4-turbo-preview", None), Some(128_000));
        assert_eq!(p.context_window("gpt-4-0613", None), Some(128_000));
        assert_eq!(p.context_window("gpt-4", None), Some(8_192));
        assert_eq!(p.context_window("gpt-3.5-turbo", None), Some(16_385));
    }

    #[test]
    fn trait_kind_and_image_tokens() {
        let p = OpenAiProvider;
        assert_eq!(LlmProvider::kind(&p), ProviderKind::OpenAi);
        assert_eq!(
            p.count_image_tokens(&ImageRef {
                width: 1024,
                height: 1024,
                high_detail: true
            }),
            765
        );
    }

    #[test]
    fn tool_role_and_tool_calls_map_to_kinds() {
        let p = OpenAiProvider;
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function"}]},
                {"role": "tool", "tool_call_id": "c1", "content": "42"}
            ]
        });
        let req = p.extract_request(&body).unwrap();
        assert_eq!(req.messages[0].content_kind, ContentKind::ToolCall);
        assert_eq!(req.messages[1].content_kind, ContentKind::ToolResult);
        assert_eq!(req.messages[1].text, "42");
    }

    #[test]
    fn content_edge_shapes() {
        let p = OpenAiProvider;
        let mk = |content| json!({"model": "gpt-4o", "messages": [{"role": "user", "content": content}]});
        let two_text = p
            .extract_request(&mk(json!([
                {"type": "text", "text": "a"}, {"type": "text", "text": "b"}
            ])))
            .unwrap();
        assert_eq!(two_text.messages[0].text, "a\nb");
        // 未知 part 类型 → 忽略
        let unknown = p
            .extract_request(&mk(json!([
                {"type": "text", "text": "x"}, {"type": "future_part", "data": 1}
            ])))
            .unwrap();
        assert_eq!(unknown.messages[0].media_count, 0);
        let img = p
            .extract_request(&mk(json!([
                {"type": "text", "text": "x"}, {"type": "input_image", "image_url": {"url": "u"}}
            ])))
            .unwrap();
        assert_eq!(img.messages[0].content_kind, ContentKind::Mixed);
        let image_only = p
            .extract_request(&mk(json!([{"type": "image_url", "image_url": {}}])))
            .unwrap();
        assert_eq!(image_only.messages[0].content_kind, ContentKind::Image);
        let null_content = json!({"model": "gpt-4o", "messages": [{"role": "assistant"}]});
        let nc = p.extract_request(&null_content).unwrap();
        assert_eq!(nc.messages[0].text, "");
        assert!(
            p.extract_request(&mk(json!(42))).is_err(),
            "非法 content 形态"
        );
        let no_role = json!({"model": "gpt-4o", "messages": [{"content": "x"}]});
        assert!(p.extract_request(&no_role).is_err(), "缺 role");
    }

    #[test]
    fn tools_flat_form_and_error_paths() {
        let p = OpenAiProvider;
        let flat = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"name": "flat_tool", "parameters": {"type": "object"}}]
        });
        let req = p.extract_request(&flat).unwrap();
        assert_eq!(req.tools[0].name, "flat_tool");
        let bad = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"parameters": {}}}]
        });
        assert!(p.extract_request(&bad).is_err(), "tools 缺 name");
        assert!(
            p.extract_response(&json!({"usage": {}})).is_err(),
            "缺 choices"
        );
    }

    #[test]
    fn stream_final_and_edge_branches() {
        let p = OpenAiProvider;
        let combined = json!({
            "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 48, "completion_tokens": 90}
        });
        assert_eq!(
            p.extract_stream_event(&combined),
            StreamEvent::Final {
                usage: Some(UsageInfo {
                    input_tokens: Some(48),
                    output_tokens: Some(90),
                    cached_tokens: None
                }),
                finish_reason: Some("stop".into())
            }
        );
        let usage_with_delta = json!({
            "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": null}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
        });
        assert!(matches!(
            p.extract_stream_event(&usage_with_delta),
            StreamEvent::Final { .. }
        ));
        let empty_usage = json!({
            "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": null}],
            "usage": {"total_tokens": 0}
        });
        assert_eq!(
            p.extract_stream_event(&empty_usage),
            StreamEvent::ContentDelta
        );
        let null_usage = json!({"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": null}], "usage": null});
        assert_eq!(
            p.extract_stream_event(&null_usage),
            StreamEvent::ContentDelta
        );
        assert_eq!(
            p.extract_stream_event(&json!({"object": "chat.completion"})),
            StreamEvent::Unrecognized
        );
    }

    #[test]
    fn inject_non_object_body_is_noop() {
        let mut not_obj = json!([1, 2, 3]);
        assert!(!inject_stream_options_include_usage(&mut not_obj));
    }

    #[test]
    fn count_tokens_covers_tools_and_messages() {
        let p = OpenAiProvider;
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello world"}
            ],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        });
        let req = p.extract_request(&body).unwrap();
        let counts = p.count_tokens(&req).unwrap();
        assert!(counts.system_tokens > 0 && counts.messages_tokens > 0 && counts.tools_tokens > 0);
    }

    #[test]
    fn extract_request_rejects_missing_model() {
        let p = OpenAiProvider;
        assert!(p.extract_request(&json!({"messages": []})).is_err());
    }
}
