//! Phase 3 接线验收（实施文档 §5 矩阵 #13 #16 本地回环）。
//!
//! 拓扑：mock 上游（axum，OS 临时端口）← proxy（session 口 OS 临时端口）
//! ← reqwest 客户端。SSE 零缓冲转发 + 末 chunk 入链（#16 核心）；
//! 绑定/回收/归属 + unattributed 桶（#13 核心）。
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use axum::routing::{get, post};
use axum::{Json, Router};
use evorule_llm_context_inspector::chain::{InspectorLog, InspectorRecord};
use evorule_llm_context_inspector::proxy::{admin_router, serve_session, ProxyState, Upstreams};
use std::sync::Arc;

async fn spawn_app(upstreams: Upstreams) -> (Arc<ProxyState>, String) {
    let chain = Arc::new(InspectorLog::in_memory());
    let state = Arc::new(ProxyState::new(chain, upstreams, 0)); // OS 临时端口（测试）
    let app = admin_router(state.clone(), Router::new());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let admin_port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (state, format!("http://127.0.0.1:{admin_port}"))
}

async fn stream_handler() -> axum::response::Response {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"he\"},\"finish_reason\":null}]}\n\n\
               data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
               data: {\"choices\":[],\"usage\":{\"prompt_tokens\":48,\"completion_tokens\":90}}\n\n";
    axum::response::Response::builder()
        .header("content-type", "text/event-stream")
        .body(axum::body::Body::from(sse))
        .unwrap()
}

async fn completions_handler(Json(_body): Json<serde_json::Value>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 4}
    }))
}

async fn spawn_mock_upstream() -> String {
    let app = Router::new()
        .route("/v1/chat/completions", post(completions_handler))
        .route("/v1/stream", post(stream_handler))
        .route("/health", get(|| async { "mock-ok" }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    format!("http://127.0.0.1:{port}")
}

#[tokio::test]
async fn acceptance_16_nonstream_roundtrip_appends_snapshot_and_observation() {
    let mock = spawn_mock_upstream().await;
    let chain = Arc::new(InspectorLog::in_memory());
    let state = Arc::new(ProxyState::new(
        chain.clone(),
        Upstreams {
            openai: mock.clone(),
            ..Upstreams::default()
        },
        0,
    ));
    let session_port = serve_session(state.clone(), 42).await.unwrap();

    // 经 session 口代理非流式请求
    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{session_port}/v1/chat/completions"
        ))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello world"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["choices"][0]["message"]["content"], "ok",
        "响应必须原样转发"
    );

    // 链上：snapshot + observation
    let records = chain.hot_records().unwrap();
    let snapshots: Vec<_> = records
        .iter()
        .filter_map(|(_, r)| match r {
            InspectorRecord::Snapshot(s) => Some(s),
            _ => None,
        })
        .collect();
    assert_eq!(snapshots.len(), 1, "请求侧一条快照");
    assert_eq!(snapshots[0].session.session_id, Some(42));
    assert_eq!(
        snapshots[0].provider,
        evorule_llm_context_inspector::ProviderKind::OpenAi
    );
    let observations: Vec<_> = records
        .iter()
        .filter_map(|(_, r)| match r {
            InspectorRecord::Observation(o) => Some(o),
            _ => None,
        })
        .collect();
    assert_eq!(observations.len(), 1, "响应侧一条观测");
    assert_eq!(observations[0].response_to, snapshots[0].snapshot_id);
    assert_eq!(
        observations[0].usage.input_tokens,
        Some(30),
        "上游真实 usage 入链"
    );
    assert!(!observations[0].stream);
    assert!(chain.verify_chain().unwrap().ok);
}

#[tokio::test]
async fn acceptance_16_sse_streams_through_and_aggregates() {
    let mock = spawn_mock_upstream().await;
    let chain = Arc::new(InspectorLog::in_memory());
    let state = Arc::new(ProxyState::new(
        chain.clone(),
        Upstreams {
            openai: mock.clone(),
            ..Upstreams::default()
        },
        0,
    ));
    let session_port = serve_session(state.clone(), 7).await.unwrap();

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{session_port}/v1/stream"))
        .header("accept", "text/event-stream")
        .body(
            serde_json::json!({
                "model": "gpt-4o",
                "stream": true,
                "messages": [{"role": "user", "content": "hi"}]
            })
            .to_string(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    assert!(ct.contains("text/event-stream"), "SSE 响应头透传");
    let text = resp.text().await.unwrap();
    assert_eq!(
        text.matches("data:").count(),
        3,
        "逐 chunk 原样转发（零缓冲不丢不并）"
    );

    // 末 chunk usage/finish 汇总为一条 stream=true 观测
    let records = chain.hot_records().unwrap();
    let observations: Vec<_> = records
        .iter()
        .filter_map(|(_, r)| match r {
            InspectorRecord::Observation(o) => Some(o),
            _ => None,
        })
        .collect();
    assert_eq!(observations.len(), 1);
    assert!(observations[0].stream);
    assert_eq!(observations[0].usage.input_tokens, Some(48));
    assert_eq!(observations[0].usage.output_tokens, Some(90));
    assert_eq!(observations[0].finish_reason.as_deref(), Some("stop"));
}

#[tokio::test]
async fn acceptance_13_bindings_and_unattributed_bucket() {
    // mock 上游对未知路由回 404（全回环，不出网）
    let mock = spawn_mock_upstream().await;
    let (state, admin) = spawn_app(Upstreams {
        openai: mock.clone(),
        ..Upstreams::default()
    })
    .await;
    let client = reqwest::Client::new();

    // 未绑定流量打 admin 口 → unattributed 桶
    let resp = client
        .post(format!("{admin}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "who am I"}]
        }))
        .send()
        .await
        .unwrap();
    // 上游不可达（默认官方端点）→ 502 也可；快照已先行入链
    let _ = resp;
    let unattr = state.list_unattributed();
    assert_eq!(unattr.len(), 1, "admin 口无绑定流量必须进 unattributed 桶");

    // 绑定会话（幂等）→ 拉起逐会话口
    let bind_resp = client
        .post(format!("{admin}/admin/bindings"))
        .json(&serde_json::json!({"session_id": 42}))
        .send()
        .await
        .unwrap();
    let status = bind_resp.status();
    let bind_text = bind_resp.text().await.unwrap();
    eprintln!("bind status={status} body={bind_text}");
    let r1: serde_json::Value = serde_json::from_str(&bind_text).unwrap();
    let port42 = r1["port"].as_u64().unwrap() as u16;
    let r2: serde_json::Value = client
        .post(format!("{admin}/admin/bindings"))
        .json(&serde_json::json!({"session_id": 42}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r2["port"].as_u64().unwrap() as u16, port42, "幂等绑定");

    // 列表与回收
    let list: serde_json::Value = client
        .get(format!("{admin}/admin/bindings"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list.as_array().unwrap().len(), 1);
    let del = client
        .delete(format!("{admin}/admin/bindings/{port42}"))
        .send()
        .await
        .unwrap();
    assert_eq!(del.status(), 204);
    assert!(!state.list_bindings().iter().any(|b| b.port == port42));
    // 二次回收 → 404
    let del2 = client
        .delete(format!("{admin}/admin/bindings/{port42}"))
        .send()
        .await
        .unwrap();
    assert_eq!(del2.status(), 404);
}

// ---- 对账（§5.3 v3.3 如实降级版）：cause_fact 存在性 + 双链校验 ----

fn fact_command(id: u64) -> evorule_reactor::Fact {
    evorule_reactor::Fact::Command {
        id: evorule_reactor::FactId(id),
        instruction: evorule_tcb::JsonValue::object_from_pairs(&[(
            "op",
            evorule_tcb::JsonValue::string("noop"),
        )]),
    }
}

#[tokio::test]
async fn acceptance_reconcile_cause_fact_existence() {
    let dir = std::env::temp_dir().join(format!("recon_test_{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).unwrap();

    // fact WAL（tier1 哈希格式，fact_hash/chain_step 用 reactor 原语）
    let fact_log = dir.join("fact.log");
    let mut prev = "genesis".to_owned();
    let mut raw_lines: Vec<String> = Vec::new();
    for (i, fact) in [fact_command(5), fact_command(6)].iter().enumerate() {
        let content_hash = evorule_reactor::fact_hash(fact).unwrap();
        let chain_hash = evorule_reactor::chain_step(&prev, &content_hash);
        let line = serde_json::json!({
            "version_before": i as u64,
            "fact": {
                "type": "Command",
                "id": if i == 0 { 5u64 } else { 6u64 },
                "instruction": {"op": "noop"}
            },
            "content_hash": content_hash,
            "prev_hash": prev,
            "chain_hash": chain_hash,
        });
        raw_lines.push(line.to_string());
        prev = chain_hash;
    }
    std::fs::write(&fact_log, raw_lines.join("\n") + "\n").unwrap();

    // inspector WAL：两条快照，cause_fact 一真一假
    let inspector_log = dir.join("inspector.log");
    {
        let chain = Arc::new(InspectorLog::open(&inspector_log).unwrap());
        let mk = |seq: u64, cause: Option<u64>| {
            use evorule_llm_context_inspector::model::{
                Composition, ContextSkeleton, ContextSnapshot, MessageEntry, SessionLink, ToolEntry,
            };
            use evorule_llm_context_inspector::{
                BindingKind, ContentKind, ProviderKind, TokenSource,
            };
            let skeleton = ContextSkeleton::new(
                ProviderKind::DeepSeek,
                "deepseek-chat",
                b"sys",
                vec![ToolEntry::new("t", b"schema")],
                vec![MessageEntry::new(
                    "user",
                    ContentKind::Text,
                    format!("m{seq}").as_bytes(),
                    0,
                )],
            );
            let composition = Composition::new(3, 5, 20, 0, 1, 1, TokenSource::Estimated).unwrap();
            ContextSnapshot::new(
                uuid::Uuid::from_u128(u128::from(seq)),
                SessionLink::new(
                    Some(1),
                    BindingKind::Header,
                    cause.map(evorule_reactor::FactId),
                ),
                seq * 1000,
                ProviderKind::DeepSeek,
                "deepseek-chat",
                None,
                composition,
                skeleton,
            )
            .unwrap()
        };
        chain
            .append(InspectorRecord::Snapshot(mk(1, Some(5))))
            .unwrap();
        chain
            .append(InspectorRecord::Snapshot(mk(2, Some(99))))
            .unwrap();
    }

    let report = evorule_llm_context_inspector::reconcile::reconcile(&fact_log, &inspector_log)
        .unwrap_or_else(|e| panic!("对账失败: {e}"));
    assert_eq!(report.facts, 2);
    assert!(report.fact_chain_hashed, "新格式 WAL → 哈希链为真校验");
    assert!(report.inspector_chain.ok);
    assert_eq!(report.cause_checked, 2);
    assert_eq!(
        report.missing_cause.len(),
        1,
        "FactId(99) 不在 fact WAL → 断引必报"
    );
    assert_eq!(report.missing_cause[0]["cause_fact"], 99);
    assert!(
        report.time_window_check.contains("unsupported"),
        "时间窗能力如实声明"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
