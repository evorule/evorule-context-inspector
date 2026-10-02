//! REST 查询端点（设计稿 §6.1 的查询面；由 admin 口合并挂载）。
//!
//! 只读面：verify / unattributed / 会话快照 / 健康摘要。
//! 管理面（/admin/bindings）在 `proxy::admin_router`。

use crate::chain::{InspectorLog, InspectorRecord};
use crate::metrics::{detect_context_events, drift_permyriad, pressure_band};
use crate::proxy::ProxyState;
use axum::extract::{Path as AxumPath, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use std::sync::Arc;

/// 查询路由（无 own state——由 admin_router 统一 with_state）。
pub fn query_router() -> Router<Arc<ProxyState>> {
    Router::new()
        .route("/api/context/verify", get(verify))
        .route("/api/context/unattributed", get(unattributed))
        .route(
            "/api/sessions/{id}/context/snapshots",
            get(session_snapshots),
        )
        .route("/api/sessions/{id}/context/health", get(session_health))
        .route(
            "/services/context_session_health",
            post(service_session_health),
        )
}

fn chain_of(state: &Arc<ProxyState>) -> Arc<InspectorLog> {
    state.chain.clone()
}

/// Plugin Contract v1.2 服务面：`POST /services/context_session_health`
/// （body={session_id}）——server 经统一 invoke 通道调用（原生优先→HTTP 回落）。
async fn service_session_health(
    State(state): State<Arc<ProxyState>>,
    axum::Json(req): axum::Json<serde_json::Value>,
) -> Response {
    let Some(sid) = req.get("session_id").and_then(serde_json::Value::as_u64) else {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            serde_json::json!({"error": "缺 session_id"}).to_string(),
        )
            .into_response();
    };
    session_health(State(state), AxumPath(sid)).await
}

async fn verify(State(state): State<Arc<ProxyState>>) -> Response {
    let report = chain_of(&state)
        .verify_chain()
        .unwrap_or_else(|e| crate::chain::ChainReport {
            records: 0,
            ok: false,
            first_bad_seq: None,
            reason: Some(e.to_string()),
        });
    Json(report).into_response()
}

async fn unattributed(State(state): State<Arc<ProxyState>>) -> Response {
    Json(state.list_unattributed()).into_response()
}

/// 会话快照摘要（热窗口内；全量以 WAL 为准——§5.2 内存治理口径）。
async fn session_snapshots(
    State(state): State<Arc<ProxyState>>,
    AxumPath(id): AxumPath<u64>,
) -> Response {
    let chain = chain_of(&state);
    let out: Vec<serde_json::Value> = match chain.hot_records() {
        Ok(records) => records
            .into_iter()
            .filter_map(|(seq, r)| match r {
                InspectorRecord::Snapshot(s) if s.session.session_id == Some(id) => {
                    Some(serde_json::json!({
                        "seq": seq,
                        "snapshot_id": s.snapshot_id,
                        "taken_at_epoch_ms": s.taken_at_epoch_ms,
                        "provider": s.provider.as_str(),
                        "model": s.model,
                        "total_tokens": s.composition.total_tokens,
                        "message_count": s.composition.message_count,
                        "token_source": s.composition.token_source,
                        "skeleton_hash": s.skeleton_hash,
                    }))
                }
                _ => None,
            })
            .collect(),
        Err(e) => {
            return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    };
    Json(out).into_response()
}

/// 健康摘要：最新快照压力 + 最近两次快照的漂移与显式检测事件（§6.1）。
async fn session_health(
    State(state): State<Arc<ProxyState>>,
    AxumPath(id): AxumPath<u64>,
) -> Response {
    let chain = chain_of(&state);
    let Ok(records) = chain.hot_records() else {
        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "链读取失败").into_response();
    };
    let snapshots: Vec<_> = records
        .into_iter()
        .filter_map(|(seq, r)| match r {
            InspectorRecord::Snapshot(s) if s.session.session_id == Some(id) => Some((seq, s)),
            _ => None,
        })
        .collect();
    let Some((last_seq, latest)) = snapshots.last().cloned() else {
        return (
            axum::http::StatusCode::NOT_FOUND,
            serde_json::json!({"error": "该会话热窗口内无快照"}).to_string(),
        )
            .into_response();
    };
    let window = latest.context_window;
    let pressure = window.map(|w| pressure_band(latest.composition.total_tokens, w));
    let (drift, events) = match snapshots.len().checked_sub(2).map(|i| &snapshots[i]) {
        Some((_, prev)) => (
            Some(drift_permyriad(&prev.skeleton, &latest.skeleton)),
            Some(detect_context_events(prev, &latest)),
        ),
        None => (None, None),
    };
    Json(serde_json::json!({
        "session_id": id,
        "last_seq": last_seq,
        "snapshot_id": latest.snapshot_id,
        "model": latest.model,
        "composition": latest.composition,
        "context_window": window,
        "pressure_band": pressure.map(|p| format!("{p:?}")),
        "drift_permyriad": drift,
        "events": events.unwrap_or_default(),
    }))
    .into_response()
}
