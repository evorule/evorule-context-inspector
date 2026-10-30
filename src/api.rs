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
/// 实载服务清单（单一事实源）：server 侧 `auto_discover` 经 `GET /services`
/// 拉取本清单装配路由；`plugin.json` 的 services 声明须与本表人工保持一致
/// （声明是装载回退，本表是身份事实源——server 侧口径）。
pub const SERVICE_NAMES: [&str; 3] = [
    "context_session_health",
    "context_verify_chain",
    "context_reconcile",
];

pub fn query_router() -> Router<Arc<ProxyState>> {
    Router::new()
        .route("/api/context/verify", get(verify))
        .route("/api/context/unattributed", get(unattributed))
        .route(
            "/api/sessions/{id}/context/snapshots",
            get(session_snapshots),
        )
        .route("/api/sessions/{id}/context/health", get(session_health))
        .route("/services", get(services))
        .route(
            "/services/context_session_health",
            post(service_session_health),
        )
        .route("/services/context_verify_chain", post(service_verify_chain))
        .route("/services/context_reconcile", post(service_reconcile))
}

/// `GET /services`：实载服务清单（server `auto_discover` 装载路径的数据源；
/// 形态契约 `{"services":[名字...]}` 与 wasm-host 同款）
async fn services() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "services": SERVICE_NAMES }))
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

/// 全链验证（含锚点流：先按节律补发，再全量验签+链接+覆盖检查）。
async fn verify(State(state): State<Arc<ProxyState>>) -> Response {
    let ctx = state.verify_context();
    let anchors = ctx.anchors.as_ref().map(|a| {
        let _ = a.publish_if_due(); // 节律补发（幂等：不足 N 条为 None）
        a.verify().unwrap_or_else(|e| crate::anchor::AnchorReport {
            entries: 0,
            ok: false,
            first_bad_seq: None,
            reason: Some(e.to_string()),
            covers_chain: false,
        })
    });
    let chain_report =
        chain_of(&state)
            .verify_chain()
            .unwrap_or_else(|e| crate::chain::ChainReport {
                records: 0,
                ok: false,
                first_bad_seq: None,
                reason: Some(e.to_string()),
            });
    Json(serde_json::json!({
        "chain": chain_report,
        "anchors": anchors,
    }))
    .into_response()
}

/// 契约服务：context_verify_chain（server invoke 通道触达；等同 /api/context/verify）。
async fn service_verify_chain(State(state): State<Arc<ProxyState>>) -> Response {
    verify(State(state)).await
}

/// 契约服务：context_reconcile（双链校验 + cause_fact 存在性对账）。
/// 路径缺省取部署配置（VerifyContext），可由调用参数覆盖。
async fn service_reconcile(
    State(state): State<Arc<ProxyState>>,
    axum::Json(req): axum::Json<serde_json::Value>,
) -> Response {
    let ctx = state.verify_context();
    let arg_path = |key: &str, default: Option<std::path::PathBuf>| -> Option<std::path::PathBuf> {
        req.get(key)
            .and_then(serde_json::Value::as_str)
            .map(std::path::PathBuf::from)
            .or(default)
    };
    let Some(fact_log) = arg_path("fact_log", ctx.fact_log_path) else {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            serde_json::json!({"error": "缺 fact_log（参数或部署配置 --fact-log）"}).to_string(),
        )
            .into_response();
    };
    let Some(inspector_log) = arg_path("inspector_log", ctx.inspector_wal_path) else {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            serde_json::json!({"error": "缺 inspector_log（仅内存模式无 WAL，不可对账）"})
                .to_string(),
        )
            .into_response();
    };
    match crate::reconcile::reconcile(&fact_log, &inspector_log) {
        Ok(report) => Json(report).into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            serde_json::json!({"error": e.to_string()}).to_string(),
        )
            .into_response(),
    }
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

#[cfg(test)]
mod services_tests {
    use super::*;

    /// 实载清单与 plugin.json 的 services 声明一致（双源防漂移:
    /// server 侧口径=本清单为身份事实源,声明仅为装载回退）
    #[test]
    fn service_names_match_plugin_json_declaration() -> Result<(), String> {
        let raw = include_str!("../plugin.json");
        let v: serde_json::Value =
            serde_json::from_str(raw).map_err(|e| format!("plugin.json 解析失败: {e}"))?;
        let declared: Vec<String> = v["services"]
            .as_array()
            .ok_or_else(|| "plugin.json 缺 services 数组".to_string())?
            .iter()
            .filter_map(|s| s["name"].as_str().map(str::to_string))
            .collect();
        let live: Vec<String> = SERVICE_NAMES.iter().map(|s| s.to_string()).collect();
        if live != declared {
            return Err(format!(
                "SERVICE_NAMES 与 plugin.json 声明漂移——新增/删除服务须两处同步: live={live:?} declared={declared:?}"
            ));
        }
        Ok(())
    }

    /// /services 响应形态与 server fetch_host_services 的解析契约一致
    /// (顶层 services 数组、元素为非空字符串)
    #[test]
    fn services_payload_matches_server_contract() {
        let body = serde_json::json!({ "services": SERVICE_NAMES });
        let Some(arr) = body.get("services").and_then(|s| s.as_array()) else {
            panic!("services 数组缺失");
        };
        assert!(!arr.is_empty());
        for item in arr {
            let n = item.as_str().unwrap_or_default();
            assert!(!n.is_empty(), "服务名不得为空");
        }
    }
}
