//! Anthropic 协议实现（Claude 系列，含 message_delta usage 路径）。
//!
//! 请求：`{model, system?: string|[{type:"text",text}], messages:
//! [{role:"user"|"assistant", content: string|parts}], tools?: [{name,
//! description?, input_schema}], max_tokens, stream?}`。
//! 响应：`{stop_reason, usage:{input_tokens, output_tokens,
//! cache_read_input_tokens?}}`。
//! SSE：`message_start`（usage.input + cache_read）→ `content_block_delta`
//! → `message_delta`（stop_reason + usage.output）→ `message_stop`。

use super::{
    ExtractedMessage, ExtractedRequest, ImageRef, LlmProvider, ResponseExtraction, StreamEvent,
    TokenCounts, ToolSchema,
};
use crate::model::UsageInfo;
use crate::provider::openai::count_with_tiktoken;
use crate::{ContentKind, InspectorError, ProviderKind};
use serde_json::Value;

/// 已知 Claude 模型窗口（保守常量表；未知 → None）。
fn builtin_context_window(model: &str) -> Option<u32> {
    if model.to_ascii_lowercase().starts_with("claude-") {
        Some(200_000)
    } else {
        None
    }
}

fn parse_anthropic_content(content: &Value) -> Result<(String, ContentKind, u32), InspectorError> {
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
                    Some("image") => media += 1,
                    Some("tool_use") | Some("tool_result") => media += 1, // 非文本块计数
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

/// Anthropic Messages API provider。
#[derive(Debug, Clone, Copy, Default)]
pub struct AnthropicProvider;

impl AnthropicProvider {
    fn system_text(body: &Value) -> Result<Option<String>, InspectorError> {
        match body.get("system") {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(Value::Array(parts)) => {
                let mut text = String::new();
                for p in parts {
                    if let Some(t) = p.get("text").and_then(Value::as_str) {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(t);
                    }
                }
                Ok(Some(text))
            }
            Some(_) => Err(InspectorError::InvalidInput("system 形态不支持".into())),
        }
    }
}

impl LlmProvider for AnthropicProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Anthropic
    }

    fn extract_request(&self, body: &Value) -> Result<ExtractedRequest, InspectorError> {
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| InspectorError::InvalidInput("缺 model".into()))?
            .to_owned();
        let system = Self::system_text(body)?;
        let arr = body
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| InspectorError::InvalidInput("缺 messages".into()))?;
        let mut messages = Vec::new();
        for (i, m) in arr.iter().enumerate() {
            let role = m
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| InspectorError::InvalidInput(format!("messages[{i}] 缺 role")))?
                .to_owned();
            let (text, kind, media) =
                parse_anthropic_content(m.get("content").unwrap_or(&Value::Null))?;
            messages.push(ExtractedMessage {
                role,
                content_kind: kind,
                text,
                media_count: media,
            });
        }
        let mut tools = Vec::new();
        if let Some(ta) = body.get("tools").and_then(Value::as_array) {
            for t in ta {
                let name = t
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| InspectorError::InvalidInput("tools[] 缺 name".into()))?
                    .to_owned();
                tools.push(ToolSchema {
                    name,
                    definition: t.clone(),
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

    fn extract_response(&self, body: &Value) -> Result<ResponseExtraction, InspectorError> {
        let usage_value = body
            .get("usage")
            .ok_or_else(|| InspectorError::InvalidInput("缺 usage".into()))?;
        let usage = UsageInfo {
            input_tokens: usage_value.get("input_tokens").and_then(Value::as_u64),
            output_tokens: usage_value.get("output_tokens").and_then(Value::as_u64),
            cached_tokens: usage_value
                .get("cache_read_input_tokens")
                .and_then(Value::as_u64),
        };
        let finish_reason = body
            .get("stop_reason")
            .and_then(Value::as_str)
            .map(str::to_owned);
        Ok(ResponseExtraction {
            usage,
            finish_reason,
        })
    }

    fn extract_stream_event(&self, chunk: &Value) -> StreamEvent {
        match chunk.get("type").and_then(Value::as_str) {
            // message_start: usage.input_tokens + cache_read_input_tokens
            Some("message_start") => {
                let u = chunk.pointer("/message/usage").unwrap_or(&Value::Null);
                StreamEvent::UsageFinal(UsageInfo {
                    input_tokens: u.get("input_tokens").and_then(Value::as_u64),
                    output_tokens: None,
                    cached_tokens: u.get("cache_read_input_tokens").and_then(Value::as_u64),
                })
            }
            // message_delta: stop_reason + usage.output_tokens 同 chunk 双字段 → Final
            Some("message_delta") => {
                let output = chunk
                    .pointer("/usage/output_tokens")
                    .and_then(Value::as_u64);
                let reason = chunk
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                if output.is_none() && reason.is_none() {
                    StreamEvent::Unrecognized
                } else {
                    StreamEvent::Final {
                        usage: Some(UsageInfo {
                            input_tokens: None,
                            output_tokens: output,
                            cached_tokens: None,
                        }),
                        finish_reason: reason,
                    }
                }
            }
            Some("content_block_delta") => StreamEvent::ContentDelta,
            _ => StreamEvent::Unrecognized,
        }
    }

    fn count_tokens(&self, req: &ExtractedRequest) -> Result<TokenCounts, InspectorError> {
        count_with_tiktoken(ProviderKind::Anthropic, req)
    }

    fn count_image_tokens(&self, image: &ImageRef) -> u64 {
        crate::tokens::estimate_image_tokens(
            ProviderKind::Anthropic,
            image.width,
            image.height,
            image.high_detail,
        )
    }

    fn context_window(&self, model: &str, _request_options: Option<&Value>) -> Option<u32> {
        builtin_context_window(model)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use serde_json::json;

    fn provider() -> AnthropicProvider {
        AnthropicProvider
    }

    #[test]
    fn context_window_known_and_unknown() {
        assert_eq!(LlmProvider::kind(&provider()), ProviderKind::Anthropic);
        assert_eq!(
            provider().context_window("claude-sonnet-4", None),
            Some(200_000)
        );
        assert_eq!(provider().context_window("gpt-4o", None), None);
    }

    #[test]
    fn system_block_array_joins_with_newline() {
        let body = json!({
            "model": "claude-sonnet-4",
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = provider().extract_request(&body).unwrap();
        assert_eq!(
            req.system.as_deref(),
            Some(
                "a
b"
            )
        );
    }

    #[test]
    fn content_array_shapes_map_to_kinds() {
        let mk = |content| json!({"model": "claude-sonnet-4", "messages": [{"role": "user", "content": content}]});
        let mixed = provider()
            .extract_request(&mk(json!([
                {"type": "text", "text": "look"}, {"type": "image", "source": {}}
            ])))
            .unwrap();
        assert_eq!(mixed.messages[0].content_kind, ContentKind::Mixed);
        assert_eq!(mixed.messages[0].media_count, 1);
        assert_eq!(mixed.messages[0].text, "look");

        let image_only = provider()
            .extract_request(&mk(json!([
                {"type": "image", "source": {}}
            ])))
            .unwrap();
        assert_eq!(image_only.messages[0].content_kind, ContentKind::Image);

        let tool_parts = provider()
            .extract_request(&mk(json!([
                {"type": "tool_use", "id": "t1", "name": "f"}
            ])))
            .unwrap();
        assert_eq!(tool_parts.messages[0].media_count, 1);
    }

    #[test]
    fn content_array_more_shapes() {
        let p = provider();
        let mk = |content| json!({"model": "claude-sonnet-4", "messages": [{"role": "user", "content": content}]});
        // 双 text part → 换行拼接
        let two = p
            .extract_request(&mk(json!([
                {"type": "text", "text": "a"}, {"type": "text", "text": "b"}
            ])))
            .unwrap();
        assert_eq!(two.messages[0].text, "a\nb");
        // 未知 part 类型 → 忽略不计数
        let unknown = p
            .extract_request(&mk(json!([
                {"type": "text", "text": "x"}, {"type": "thinking", "thinking": "..."}
            ])))
            .unwrap();
        assert_eq!(unknown.messages[0].media_count, 0);
        // content 缺省（Null）→ 空文本
        let nc = p
            .extract_request(
                &json!({"model": "claude-sonnet-4", "messages": [{"role": "assistant"}]}),
            )
            .unwrap();
        assert_eq!(nc.messages[0].text, "");
        // 非法 content 形态 → 拒收
        assert!(p.extract_request(&mk(json!(42))).is_err());
    }

    #[test]
    fn extract_request_error_paths() {
        assert!(
            provider()
                .extract_request(&json!({"messages": []}))
                .is_err(),
            "缺 model"
        );
        assert!(
            provider()
                .extract_request(&json!({"model": "claude-sonnet-4"}))
                .is_err(),
            "缺 messages"
        );
        let bad_tool = json!({
            "model": "claude-sonnet-4",
            "messages": [],
            "tools": [{"description": "no name"}]
        });
        assert!(
            provider().extract_request(&bad_tool).is_err(),
            "tools 缺 name"
        );
        let bad_system = json!({"model": "claude-sonnet-4", "system": 42, "messages": []});
        assert!(
            provider().extract_request(&bad_system).is_err(),
            "system 形态不支持"
        );
    }

    #[test]
    fn extract_response_requires_usage() {
        assert!(provider()
            .extract_response(&json!({"stop_reason": "end_turn"}))
            .is_err());
    }

    #[test]
    fn stream_event_branches() {
        let p = provider();
        let start = json!({"type": "message_start", "message": {"usage": {"input_tokens": 9, "cache_read_input_tokens": 3}}});
        assert_eq!(
            p.extract_stream_event(&start),
            StreamEvent::UsageFinal(UsageInfo {
                input_tokens: Some(9),
                output_tokens: None,
                cached_tokens: Some(3)
            })
        );
        let delta =
            json!({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}});
        assert_eq!(p.extract_stream_event(&delta), StreamEvent::ContentDelta);
        let stop_only = json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}});
        assert_eq!(
            p.extract_stream_event(&stop_only),
            StreamEvent::Final {
                usage: Some(UsageInfo {
                    input_tokens: None,
                    output_tokens: None,
                    cached_tokens: None
                }),
                finish_reason: Some("end_turn".into())
            }
        );
        let empty_delta = json!({"type": "message_delta"});
        assert_eq!(
            p.extract_stream_event(&empty_delta),
            StreamEvent::Unrecognized
        );
        assert_eq!(
            p.extract_stream_event(&json!({"type": "ping"})),
            StreamEvent::Unrecognized
        );
    }

    #[test]
    fn count_tokens_and_images() {
        let body = json!({
            "model": "claude-sonnet-4",
            "system": "sys",
            "messages": [{"role": "user", "content": "hello world"}],
            "tools": [{"name": "f", "input_schema": {"type": "object"}}]
        });
        let req = provider().extract_request(&body).unwrap();
        let counts = provider().count_tokens(&req).unwrap();
        assert!(counts.system_tokens > 0 && counts.messages_tokens > 0 && counts.tools_tokens > 0);
        assert_eq!(
            provider().count_image_tokens(&ImageRef {
                width: 1092,
                height: 1092,
                high_detail: true
            }),
            1590
        );
    }
}
