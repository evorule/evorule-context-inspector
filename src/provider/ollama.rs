//! Ollama 协议实现（本地模型；估算计数 + `options.num_ctx` 窗口解析）。
//!
//! 请求：`{model, messages, stream?, options?{num_ctx}}`（/api/chat）。
//! 响应（stream=false）：`{message, done, prompt_eval_count, eval_count,
//! done_reason?}`。
//! 流式：NDJSON 行，中间 `{message:{content}, done:false}`，末行
//! `{done:true, prompt_eval_count, eval_count, done_reason:"length"|"stop"}`。
//! usage 映射：`prompt_eval_count` → input，`eval_count` → output
//! （字段缺失为 None）；`done_reason:"length"` → 截断信号。

use super::openai::count_with_tiktoken;
use super::{
    ExtractedMessage, ExtractedRequest, ImageRef, LlmProvider, ResponseExtraction, StreamEvent,
    TokenCounts,
};
use crate::model::UsageInfo;
use crate::{ContentKind, InspectorError, ProviderKind};
use serde_json::Value;

/// Ollama 服务端默认窗口（§2.6：options.num_ctx 或默认 4096）。
pub const OLLAMA_DEFAULT_NUM_CTX: u32 = 4096;

fn parse_ollama_content(content: &Value) -> (String, ContentKind, u32) {
    match content {
        Value::String(s) => (s.clone(), ContentKind::Text, 0),
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
            (text, kind, media)
        }
        _ => (String::new(), ContentKind::Text, 0),
    }
}

/// Ollama /api/chat provider。
#[derive(Debug, Clone, Copy, Default)]
pub struct OllamaProvider;

impl LlmProvider for OllamaProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Ollama
    }

    fn extract_request(&self, body: &Value) -> Result<ExtractedRequest, InspectorError> {
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| InspectorError::InvalidInput("缺 model".into()))?
            .to_owned();
        let arr = body
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| InspectorError::InvalidInput("缺 messages".into()))?;
        let mut system_parts: Vec<String> = Vec::new();
        let mut messages = Vec::new();
        for (i, m) in arr.iter().enumerate() {
            let role = m
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| InspectorError::InvalidInput(format!("messages[{i}] 缺 role")))?
                .to_owned();
            let (text, kind, media) =
                parse_ollama_content(m.get("content").unwrap_or(&Value::Null));
            if role == "system" {
                system_parts.push(text);
            } else {
                messages.push(ExtractedMessage {
                    role,
                    content_kind: kind,
                    text,
                    media_count: media,
                });
            }
        }
        Ok(ExtractedRequest {
            model,
            system: if system_parts.is_empty() {
                None
            } else {
                Some(system_parts.join("\n"))
            },
            tools: Vec::new(), // /api/chat 工具面当前不含 schema 声明（按需扩展）
            messages,
            request_options: body.get("options").cloned(),
            raw: body.clone(),
        })
    }

    fn extract_response(&self, body: &Value) -> Result<ResponseExtraction, InspectorError> {
        Ok(ResponseExtraction {
            usage: UsageInfo {
                input_tokens: body.get("prompt_eval_count").and_then(Value::as_u64),
                output_tokens: body.get("eval_count").and_then(Value::as_u64),
                cached_tokens: None, // Ollama 无 prompt cache 计数字段（§3.2 → 指标 None）
            },
            finish_reason: body
                .get("done_reason")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
    }

    fn extract_stream_event(&self, chunk: &Value) -> StreamEvent {
        match chunk.get("done").and_then(Value::as_bool) {
            Some(true) => StreamEvent::Final {
                usage: Some(UsageInfo {
                    input_tokens: chunk.get("prompt_eval_count").and_then(Value::as_u64),
                    output_tokens: chunk.get("eval_count").and_then(Value::as_u64),
                    cached_tokens: None,
                }),
                finish_reason: chunk
                    .get("done_reason")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            },
            Some(false) => StreamEvent::ContentDelta,
            None => StreamEvent::Unrecognized,
        }
    }

    fn count_tokens(&self, req: &ExtractedRequest) -> Result<TokenCounts, InspectorError> {
        count_with_tiktoken(ProviderKind::Ollama, req)
    }

    fn count_image_tokens(&self, _image: &ImageRef) -> u64 {
        0
    }

    /// `options.num_ctx` 优先，否则服务端默认 4096（§2.6：本地可配，须读实际值）。
    fn context_window(&self, _model: &str, request_options: Option<&Value>) -> Option<u32> {
        let num_ctx = request_options
            .and_then(|o| o.get("num_ctx"))
            .and_then(Value::as_u64);
        Some(num_ctx.map(|n| n as u32).unwrap_or(OLLAMA_DEFAULT_NUM_CTX))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use serde_json::json;

    #[test]
    fn num_ctx_takes_precedence_with_default_fallback() {
        let p = OllamaProvider;
        assert_eq!(LlmProvider::kind(&p), ProviderKind::Ollama);
        assert_eq!(
            p.context_window("llama3", Some(&json!({"num_ctx": 8192}))),
            Some(8192)
        );
        assert_eq!(p.context_window("llama3", Some(&json!({}))), Some(4096));
        assert_eq!(p.context_window("llama3", None), Some(4096));
    }

    #[test]
    fn ollama_response_maps_counts() {
        let p = OllamaProvider;
        let body = json!({
            "message": {"role": "assistant", "content": "ok"},
            "done": true, "prompt_eval_count": 25, "eval_count": 8, "done_reason": "stop"
        });
        let ex = p.extract_response(&body).unwrap();
        assert_eq!(ex.usage.input_tokens, Some(25));
        assert_eq!(ex.usage.output_tokens, Some(8));
        assert_eq!(ex.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn ollama_stream_final_line_aggregates() {
        let p = OllamaProvider;
        let mut agg = crate::provider::StreamAggregator::new();
        agg.observe(p.extract_stream_event(&json!({"message": {"content": "he"}, "done": false})));
        agg.observe(p.extract_stream_event(&json!({"message": {"content": "llo"}, "done": false})));
        agg.observe(p.extract_stream_event(&json!({"done": true, "prompt_eval_count": 25, "eval_count": 8, "done_reason": "length"})));
        let out = agg.finish();
        assert_eq!(out.content_chunks, 2);
        assert_eq!(out.usage.input_tokens, Some(25));
        assert_eq!(out.usage.output_tokens, Some(8));
        assert_eq!(out.finish_reason.as_deref(), Some("length"));
    }
}

#[cfg(test)]
mod extract_request_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_request_captures_options_and_system() {
        let p = OllamaProvider;
        let body = json!({
            "model": "llama3.2",
            "options": {"num_ctx": 8192, "temperature": 0},
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image", "source": {}}]}
            ]
        });
        let req = p.extract_request(&body).unwrap();
        assert_eq!(req.system.as_deref(), Some("s"));
        assert_eq!(
            req.request_options
                .as_ref()
                .and_then(|o| o.get("num_ctx"))
                .and_then(Value::as_u64),
            Some(8192)
        );
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].content_kind, ContentKind::Mixed);
    }

    #[test]
    fn extract_request_error_paths() {
        let p = OllamaProvider;
        assert!(
            p.extract_request(&json!({"messages": []})).is_err(),
            "缺 model"
        );
        assert!(
            p.extract_request(&json!({"model": "llama3.2"})).is_err(),
            "缺 messages"
        );
    }

    #[test]
    fn ollama_content_more_shapes() {
        let p = OllamaProvider;
        let mk = |content| json!({"model": "llama3.2", "messages": [{"role": "user", "content": content}]});
        let two = p
            .extract_request(&mk(json!([
                {"type": "text", "text": "a"}, {"type": "text", "text": "b"}
            ])))
            .unwrap();
        assert_eq!(two.messages[0].text, "a\nb");
        let unknown = p
            .extract_request(&mk(json!([
                {"type": "text", "text": "x"}, {"type": "weird", "data": 1}
            ])))
            .unwrap();
        assert_eq!(unknown.messages[0].media_count, 0);
        let image_only = p
            .extract_request(&mk(json!([{"type": "image", "source": {}}])))
            .unwrap();
        assert_eq!(image_only.messages[0].content_kind, ContentKind::Image);
        let empty_arr = p.extract_request(&mk(json!([]))).unwrap();
        assert_eq!(empty_arr.messages[0].content_kind, ContentKind::Text);
        // 非法 content 形态 → 空文本不炸（Ollama 宽容语义）
        let bad = p.extract_request(&mk(json!(42))).unwrap();
        assert_eq!(bad.messages[0].text, "");
        // content 缺省
        let nc = p
            .extract_request(&json!({"model": "llama3.2", "messages": [{"role": "assistant"}]}))
            .unwrap();
        assert_eq!(nc.messages[0].text, "");
    }

    #[test]
    fn count_and_stream_misc() {
        let p = OllamaProvider;
        let body = json!({
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "hello world"}]
        });
        let req = p.extract_request(&body).unwrap();
        let counts = p.count_tokens(&req).unwrap();
        assert!(counts.messages_tokens > 0);
        assert_eq!(
            p.count_image_tokens(&ImageRef {
                width: 10,
                height: 10,
                high_detail: true
            }),
            0
        );
        assert_eq!(
            p.extract_stream_event(&json!({"other": 1})),
            StreamEvent::Unrecognized
        );
    }
}
