//! Phase 2 验收测试（实施文档 §5 矩阵 #1 #2 #3 #4 #8 #9）。
//!
//! fixture 来源两档：真实录制（MiniMax OpenAI 兼容端点，2026-10-02）与
//! spec 派生（无凭据 provider，见 tests/fixtures/README.md）。
//! `--store-content` / 网络层（MITM 转发）不在本文件 —— 属 Phase 3。
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use evorule_llm_context_inspector::model::{
    Composition, ContextSkeleton, ContextSnapshot, ResponseObservation, SessionLink, UsageInfo,
};
use evorule_llm_context_inspector::provider::{
    anthropic::AnthropicProvider,
    deepseek::DeepSeekProvider,
    ollama::OllamaProvider,
    openai::{inject_stream_options_include_usage, OpenAiProvider},
    LlmProvider, StreamAggregator,
};
use evorule_llm_context_inspector::{metrics, BindingKind, ProviderKind, TokenSource};
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

fn fixture(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("读 fixture 失败 {path:?}: {e}"))
}

fn fixture_json(name: &str) -> Value {
    serde_json::from_str(&fixture(name)).unwrap_or_else(|e| panic!("fixture {name} JSON 非法: {e}"))
}

/// SSE 文本 → chunk JSON 序列（跳过 `data: [DONE]` 与非 data 行）。
fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|l| *l != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("SSE 行 JSON 非法: {e}")))
        .collect()
}

/// NDJSON 文本 → 行 JSON 序列（Ollama 流式）。
fn ndjson_lines(text: &str) -> Vec<Value> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("NDJSON 行 JSON 非法: {e}")))
        .collect()
}

static SEQ: AtomicU64 = AtomicU64::new(1);

/// 抽取结果 → 骨架 + 组成 → 快照（#1 的「生成 Snapshot」全路径）。
fn build_snapshot(
    provider: ProviderKind,
    extracted: &evorule_llm_context_inspector::provider::ExtractedRequest,
    counts: &evorule_llm_context_inspector::provider::TokenCounts,
) -> ContextSnapshot {
    let seq = SEQ.fetch_add(1, Ordering::SeqCst);
    let skeleton = ContextSkeleton::new(
        provider,
        extracted.model.clone(),
        extracted.system.as_deref().unwrap_or_default().as_bytes(),
        evorule_llm_context_inspector::provider::skeleton_tools(extracted).unwrap_or_default(),
        extracted
            .messages
            .iter()
            .map(|m| {
                evorule_llm_context_inspector::model::MessageEntry::new(
                    m.role.clone(),
                    m.content_kind,
                    m.text.as_bytes(),
                    m.media_count,
                )
            })
            .collect(),
    );
    let composition = Composition::with_total(
        counts.system_tokens,
        counts.tools_tokens,
        counts.messages_tokens,
        counts.images_tokens,
        counts.system_tokens + counts.tools_tokens + counts.messages_tokens + counts.images_tokens,
        u32::try_from(extracted.messages.len()).unwrap_or(u32::MAX),
        u32::try_from(extracted.tools.len()).unwrap_or(u32::MAX),
        TokenSource::Estimated,
    )
    .unwrap_or_else(|e| panic!("组成核账失败: {e}"));
    let window = evorule_llm_context_inspector::provider::window_of(
        provider,
        &extracted.model,
        extracted.request_options.as_ref(),
    );
    ContextSnapshot::new(
        Uuid::new_v4(),
        SessionLink::new(None, BindingKind::Unattributed, None),
        1_760_000_000_000 + seq,
        provider,
        extracted.model.clone(),
        window,
        composition,
        skeleton,
    )
    .unwrap_or_else(|e| panic!("快照构造失败: {e}"))
}

// ---- 验收 #1：OpenAI 请求/响应 → Snapshot（真实录制 fixture）----

#[test]
fn acceptance_1_openai_request_response_to_snapshot() {
    let p = OpenAiProvider;
    let mut body = fixture_json("openai_request.json");
    // §4.3：流式路径先注入 stream_options，注入后才允许算骨架
    assert!(inject_stream_options_include_usage(&mut body));
    let extracted = p
        .extract_request(&body)
        .unwrap_or_else(|e| panic!("抽取失败: {e}"));
    assert_eq!(
        extracted.system.as_deref(),
        Some("You are a terse assistant.")
    );
    assert_eq!(extracted.messages.len(), 1);

    let counts = p
        .count_tokens(&extracted)
        .unwrap_or_else(|e| panic!("计数失败: {e}"));
    assert!(
        counts.messages_tokens > 0 && counts.system_tokens > 0,
        "估算应非零"
    );

    let mut snap = build_snapshot(ProviderKind::OpenAi, &extracted, &counts);
    assert_eq!(
        snap.skeleton_hash,
        snap.skeleton.skeleton_hash().unwrap_or_default()
    );

    // 真实响应 → usage 回填
    let resp = fixture_json("openai_nonstream.json");
    let ex = p
        .extract_response(&resp)
        .unwrap_or_else(|e| panic!("响应抽取失败: {e}"));
    let obs = ResponseObservation::new(
        Uuid::new_v4(),
        snap.snapshot_id,
        1_760_000_001_000,
        ex.finish_reason,
        ex.usage,
        metrics::detect_truncation(ProviderKind::OpenAi, Some("stop"), None, None),
        false,
    );
    assert_eq!(
        obs.usage.input_tokens,
        Some(28),
        "真实 usage 必须原样保留（证据锚）"
    );
    assert!(snap.backfill_token_source(&obs).unwrap_or(false));
    assert_eq!(snap.composition.token_source, TokenSource::ProviderReported);
}

// ---- 验收 #2：Anthropic（spec 派生 fixture）----

#[test]
fn acceptance_2_anthropic_request_response() {
    let p = AnthropicProvider;
    let body = fixture_json("anthropic_request.json");
    let extracted = p
        .extract_request(&body)
        .unwrap_or_else(|e| panic!("抽取失败: {e}"));
    assert_eq!(
        extracted.system.as_deref(),
        Some("You are a terse assistant.")
    );
    assert_eq!(extracted.tools.len(), 1);
    assert_eq!(extracted.tools[0].name, "get_weather");

    let resp = fixture_json("anthropic_nonstream.json");
    let ex = p
        .extract_response(&resp)
        .unwrap_or_else(|e| panic!("响应抽取失败: {e}"));
    assert_eq!(ex.usage.input_tokens, Some(42));
    assert_eq!(
        ex.usage.cached_tokens,
        Some(20),
        "cache_read_input_tokens 必须映射"
    );
    assert_eq!(ex.finish_reason.as_deref(), Some("end_turn"));
    assert!(!metrics::detect_truncation(
        ProviderKind::Anthropic,
        ex.finish_reason.as_deref(),
        None,
        None
    ));
}

// ---- 验收 #3：DeepSeek（spec 派生 fixture，含 cache 字段）----

#[test]
fn acceptance_3_deepseek_cache_fields() {
    let p = DeepSeekProvider;
    let body = fixture_json("deepseek_request.json");
    let extracted = p
        .extract_request(&body)
        .unwrap_or_else(|e| panic!("抽取失败: {e}"));
    assert_eq!(extracted.model, "deepseek-chat");

    let resp = fixture_json("deepseek_nonstream.json");
    let ex = p
        .extract_response(&resp)
        .unwrap_or_else(|e| panic!("响应抽取失败: {e}"));
    assert_eq!(
        ex.usage.cached_tokens,
        Some(768),
        "prompt_cache_hit_tokens 必须映射"
    );
    assert_eq!(ex.usage.input_tokens, Some(1033));
}

// ---- 验收 #4：SSE 流式 × 3 provider（逐 chunk + 末 chunk 汇总）----

#[test]
fn acceptance_4_openai_stream_real_fixture() {
    let p = OpenAiProvider;
    let mut agg = StreamAggregator::new();
    for chunk in sse_chunks(&fixture("openai_stream.sse")) {
        agg.observe(p.extract_stream_event(&chunk));
    }
    let out = agg.finish();
    assert!(
        out.content_chunks > 0,
        "真实流必有内容 chunk（逐 chunk 观测证据）"
    );
    assert_eq!(out.usage.input_tokens, Some(48), "真实末 chunk usage");
    assert_eq!(out.usage.output_tokens, Some(90));
}

#[test]
fn acceptance_4_anthropic_stream_message_delta_dual_field() {
    let p = AnthropicProvider;
    let mut agg = StreamAggregator::new();
    for chunk in sse_chunks(&fixture("anthropic_stream.sse")) {
        agg.observe(p.extract_stream_event(&chunk));
    }
    let out = agg.finish();
    assert!(out.content_chunks > 0);
    assert_eq!(out.usage.input_tokens, Some(42), "message_start 的 input");
    assert_eq!(out.usage.cached_tokens, Some(20));
    assert_eq!(
        out.usage.output_tokens,
        Some(64),
        "message_delta 的 output（字段合并不覆盖）"
    );
    assert_eq!(out.finish_reason.as_deref(), Some("max_tokens"));
    assert!(
        metrics::detect_truncation(
            ProviderKind::Anthropic,
            out.finish_reason.as_deref(),
            None,
            None
        ),
        "max_tokens 结束必须判截断"
    );
}

#[test]
fn acceptance_4_deepseek_and_ollama_streams() {
    let ds = DeepSeekProvider;
    let mut agg = StreamAggregator::new();
    for chunk in sse_chunks(&fixture("deepseek_stream.sse")) {
        agg.observe(ds.extract_stream_event(&chunk));
    }
    let out = agg.finish();
    assert_eq!(out.usage.cached_tokens, Some(64), "流式 cache 字段");
    assert_eq!(out.finish_reason.as_deref(), Some("stop"));

    let ol = OllamaProvider;
    let mut agg2 = StreamAggregator::new();
    for line in ndjson_lines(&fixture("ollama_stream.ndjson")) {
        agg2.observe(ol.extract_stream_event(&line));
    }
    let out2 = agg2.finish();
    assert_eq!(out2.usage.input_tokens, Some(25));
    assert_eq!(out2.finish_reason.as_deref(), Some("length"));
    assert!(metrics::detect_truncation(
        ProviderKind::Ollama,
        out2.finish_reason.as_deref(),
        None,
        None
    ));
}

// ---- 验收 #8：重复率（对照手算 headline；完整用例在 metrics 单测）----

#[test]
fn acceptance_8_duplicate_ratio_headline() {
    // 显式 N=1：全重叠 → 10000（大小写归一）
    let two = vec!["a b c d e f g h".to_owned(), "A B C D E F G H".to_owned()];
    assert_eq!(metrics::duplicate_ratio_permyriad(&two, 1, 8), 10_000);
    // 默认 N=5：earlier=[base,novelA]，recent 5 条含 base → 交集 1/5 = 2000
    let seven = vec![
        "a b c d e f g h".to_owned(),
        "i j k l m n o p".to_owned(),
        "a b c d e f g h".to_owned(),
        "q1 q2 q3 q4 q5 q6 q7 q8".to_owned(),
        "r1 r2 r3 r4 r5 r6 r7 r8".to_owned(),
        "s1 s2 s3 s4 s5 s6 s7 s8".to_owned(),
        "t1 t2 t3 t4 t5 t6 t7 t8".to_owned(),
    ];
    assert_eq!(metrics::duplicate_ratio_permyriad_default(&seven), 2_000);
    // 消息不足 N 条 → 全部归 recent，earlier 为空 → 0
    assert_eq!(
        metrics::duplicate_ratio_permyriad_default(&["a b c".to_owned()]),
        0
    );
}

// ---- 验收 #9：截断检测（headline；参数化全表在 metrics 单测）----

#[test]
fn acceptance_9_truncation_headline() {
    assert!(metrics::detect_truncation(
        ProviderKind::OpenAi,
        Some("length"),
        None,
        None
    ));
    assert!(metrics::detect_truncation(
        ProviderKind::Anthropic,
        Some("max_tokens"),
        None,
        None
    ));
    assert!(!metrics::detect_truncation(
        ProviderKind::OpenAi,
        Some("stop"),
        None,
        None
    ));
    assert!(metrics::detect_truncation(
        ProviderKind::OpenAi,
        Some("stop"),
        Some("... [truncated]"),
        None
    ));
}

// ---- UsageInfo 直引用防失效（真实 usage 形态对齐证据）----

#[test]
fn usage_info_shape_is_the_recorded_one() {
    let u = UsageInfo {
        input_tokens: Some(28),
        output_tokens: Some(142),
        cached_tokens: None,
    };
    assert_eq!(u.input_tokens, Some(28));
}
