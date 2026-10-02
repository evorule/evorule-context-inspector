//! DeepSeek 协议实现 —— 组合复用 OpenAI 兼容路径，扩展专有 cache 字段
//! （`usage.prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`）。
//! 实施文档 v3 §1 决策 10：4 实现口径，DeepSeek 复用 OpenAI 抽取逻辑。

use super::openai::{count_with_tiktoken, parse_usage, OpenAiProvider};
use super::{
    ExtractedRequest, ImageRef, LlmProvider, ResponseExtraction, StreamEvent, TokenCounts,
};
use crate::model::UsageInfo;
use crate::{InspectorError, ProviderKind};
use serde_json::Value;

/// 已知 DeepSeek 模型窗口（保守常量表）。
fn builtin_context_window(model: &str) -> Option<u32> {
    let m = model.to_ascii_lowercase();
    if m.starts_with("deepseek-") {
        // chat / v3 / reasoner / r1 现役全系 128k（保守常量表）
        Some(128_000)
    } else {
        None
    }
}

/// DeepSeek provider（抽取逻辑组合自 OpenAI 兼容实现，cache 字段扩展）。
#[derive(Debug, Clone, Copy, Default)]
pub struct DeepSeekProvider;

impl DeepSeekProvider {
    /// OpenAI usage 块 + DeepSeek 专有显式 cache 字段。
    fn parse_usage_with_cache(u: &Value) -> UsageInfo {
        let mut usage = parse_usage(u);
        if let Some(hit) = u.get("prompt_cache_hit_tokens").and_then(Value::as_u64) {
            usage.cached_tokens = Some(hit);
        }
        usage
    }
}

impl LlmProvider for DeepSeekProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::DeepSeek
    }

    fn extract_request(&self, request_body: &Value) -> Result<ExtractedRequest, InspectorError> {
        OpenAiProvider.extract_request(request_body)
    }

    fn extract_response(
        &self,
        response_body: &Value,
    ) -> Result<ResponseExtraction, InspectorError> {
        let mut ex = OpenAiProvider.extract_response(response_body)?;
        if let Some(u) = response_body.get("usage") {
            ex.usage = Self::parse_usage_with_cache(u);
        }
        Ok(ex)
    }

    fn extract_stream_event(&self, chunk: &Value) -> StreamEvent {
        // usage chunk（空 choices）→ 带 cache 字段解析
        if chunk
            .get("choices")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
        {
            if let Some(u) = chunk.get("usage") {
                return StreamEvent::UsageFinal(Self::parse_usage_with_cache(u));
            }
        }
        OpenAiProvider.extract_stream_event(chunk)
    }

    fn count_tokens(&self, req: &ExtractedRequest) -> Result<TokenCounts, InspectorError> {
        count_with_tiktoken(ProviderKind::DeepSeek, req)
    }

    fn count_image_tokens(&self, image: &ImageRef) -> u64 {
        // DeepSeek 文本模型暂无公开图片计费口径 → 0（不计）
        let _ = image;
        0
    }

    fn context_window(&self, model: &str, _request_options: Option<&Value>) -> Option<u32> {
        builtin_context_window(model)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::provider::StreamAggregator;
    use serde_json::json;

    #[test]
    fn deepseek_response_maps_explicit_cache_field() {
        let p = DeepSeekProvider;
        let body = json!({
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 1033, "completion_tokens": 199,
                "prompt_cache_hit_tokens": 768, "prompt_cache_miss_tokens": 265
            }
        });
        let ex = p.extract_response(&body).unwrap();
        assert_eq!(ex.usage.input_tokens, Some(1033));
        assert_eq!(
            ex.usage.cached_tokens,
            Some(768),
            "显式 cache 字段必须映射到 cached_tokens"
        );
        assert_eq!(ex.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn deepseek_stream_usage_chunk_carries_cache() {
        let p = DeepSeekProvider;
        let mut agg = StreamAggregator::new();
        for line in [
            json!({"choices": [{"delta": {"content": "ok"}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_cache_hit_tokens": 64}}),
        ] {
            agg.observe(p.extract_stream_event(&line));
        }
        let out = agg.finish();
        assert_eq!(out.usage.input_tokens, Some(100));
        assert_eq!(out.usage.cached_tokens, Some(64));
        assert_eq!(out.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn deepseek_window_and_stream_passthrough() {
        let p = DeepSeekProvider;
        assert_eq!(p.context_window("deepseek-reasoner", None), Some(128_000));
        assert_eq!(p.context_window("mystery", None), None);
        // 空 choices 且无 usage → 透传识别
        assert_eq!(
            p.extract_stream_event(&json!({"choices": []})),
            crate::provider::StreamEvent::Unrecognized
        );
        // 非 usage chunk 走 OpenAI 兼容路径
        assert_eq!(
            p.extract_stream_event(
                &json!({"choices": [{"delta": {"content": "x"}, "finish_reason": null}]})
            ),
            crate::provider::StreamEvent::ContentDelta
        );
        assert_eq!(
            p.extract_stream_event(&json!({"unrelated": true})),
            crate::provider::StreamEvent::Unrecognized
        );
        let body = json!({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello world"}],
            "tools": [{"type": "function", "function": {"name": "f"}}]
        });
        let req = p.extract_request(&body).unwrap();
        let counts = p.count_tokens(&req).unwrap();
        assert!(counts.messages_tokens > 0 && counts.tools_tokens > 0);
    }

    #[test]
    fn deepseek_kind_and_image_semantics() {
        let p = DeepSeekProvider;
        assert_eq!(LlmProvider::kind(&p), ProviderKind::DeepSeek);
        assert_eq!(
            p.count_image_tokens(&ImageRef {
                width: 100,
                height: 100,
                high_detail: true
            }),
            0
        );
    }

    #[test]
    fn deepseek_request_reuses_openai_path() {
        let p = DeepSeekProvider;
        let body = json!({
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            "tools": [{"type": "function", "function": {"name": "calc"}}]
        });
        let req = p.extract_request(&body).unwrap();
        assert_eq!(req.model, "deepseek-chat");
        assert_eq!(req.tools[0].name, "calc");
        assert_eq!(p.context_window("deepseek-chat", None), Some(128_000));
    }
}
