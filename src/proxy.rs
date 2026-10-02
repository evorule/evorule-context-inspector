//! in-process MITM 监听（设计稿 §4.1/§4.2/§4.3，v3.1 插件形态）。
//!
//! - **逐会话端口**：`--session-port-base`（默认 19101）起递增；0 = OS 临时
//!   端口（测试用）。agent 只改 base_url，零业务代码（§4.2 默认推荐）。
//! - **统一入口（admin 口，默认 19100）**：管理面（`/admin/bindings`、查询
//!   端点、`/health`）+ 代理 fallback——无绑定流量进 unattributed 桶
//!   （ContextSessionUnknown，失败可见不静默，§4.2 ③）。
//! - **SSE 零缓冲**：上游 `bytes_stream` 逐 chunk 经 mpsc 通道转发（首字
//!   延迟不增），转发任务内同步切行喂聚合器，流结束后以 `stream:true` 落
//!   一条 `ResponseObservation`（§4.3）。OpenAI 流式请求在透传前注入
//!   `stream_options.include_usage`，骨架以**改写后**请求计算（哈希=实发）。
//! - **Header 纪律**：Authorization 等只透传，永不入日志/哈希/落盘（§2.5）；
//!   逐跳头按 RFC 7230 剥离。

use crate::chain::{InspectorLog, InspectorRecord};
use crate::model::{
    Composition, ContextSkeleton, ContextSnapshot, MessageEntry, ResponseObservation, SessionLink,
};
use crate::provider::openai::inject_stream_options_include_usage;
use crate::provider::{
    anthropic::AnthropicProvider, deepseek::DeepSeekProvider, ollama::OllamaProvider,
    openai::OpenAiProvider, LlmProvider, StreamAggregator,
};
use crate::{BindingKind, ProviderKind, SessionId};
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::Router;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// 会话绑定（端口 → 会话）。
#[derive(Debug, Clone, Serialize)]
pub struct SessionBinding {
    pub session_id: SessionId,
    pub port: u16,
    pub bound_at_epoch_ms: u64,
}

/// 未识别流量桶条目（失败可见不静默）。
#[derive(Debug, Clone, Serialize)]
pub struct UnattributedEntry {
    pub snapshot_id: uuid::Uuid,
    pub taken_at_epoch_ms: u64,
    pub model: String,
}

/// 上游地址（可由启动参数覆盖；默认官方端点 / 本机 Ollama）。
#[derive(Debug, Clone)]
pub struct Upstreams {
    pub openai: String,
    pub anthropic: String,
    pub deepseek: String,
    pub ollama: String,
}

impl Default for Upstreams {
    fn default() -> Self {
        Self {
            openai: "https://api.openai.com".into(),
            anthropic: "https://api.anthropic.com".into(),
            deepseek: "https://api.deepseek.com".into(),
            ollama: "http://127.0.0.1:11434".into(),
        }
    }
}

/// 验证上下文（锚点服务 + 对账路径；启动期一次性注入）。
#[derive(Default, Clone)]
pub struct VerifyContext {
    pub anchors: Option<Arc<crate::anchor::AnchorService>>,
    pub inspector_wal_path: Option<std::path::PathBuf>,
    pub fact_log_path: Option<std::path::PathBuf>,
}

/// 代理共享状态（admin 口与所有会话口共用）。
pub struct ProxyState {
    /// 链日志（落盘或内存，由启动参数决定）。
    pub chain: Arc<InspectorLog>,
    pub upstreams: Upstreams,
    verify: std::sync::OnceLock<VerifyContext>,
    /// `--store-content` 旁挂目录（Some = 启用原文旁挂；None = 默认强隐私）。
    content_dir: std::sync::OnceLock<Option<std::path::PathBuf>>,
    http: reqwest::Client,
    bindings: RwLock<HashMap<u16, SessionBinding>>,
    /// 已拉起监听器的端口（幂等 serve 判定）。
    served: RwLock<std::collections::HashSet<u16>>,
    unattributed: RwLock<Vec<UnattributedEntry>>,
    next_port: AtomicU16,
    session_port_base: u16,
}

impl ProxyState {
    pub fn new(chain: Arc<InspectorLog>, upstreams: Upstreams, session_port_base: u16) -> Self {
        Self {
            chain,
            upstreams,
            verify: std::sync::OnceLock::new(),
            content_dir: std::sync::OnceLock::new(),
            http: reqwest::Client::new(),
            bindings: RwLock::new(HashMap::new()),
            served: RwLock::new(std::collections::HashSet::new()),
            unattributed: RwLock::new(Vec::new()),
            next_port: AtomicU16::new(session_port_base),
            session_port_base,
        }
    }

    /// 绑定会话 → 占位端口（§4.2）。已绑定会话幂等返回原端口。
    /// 实际端口以 `serve_session` 绑定监听后的回填为准（`rebind_port`）。
    pub fn bind_session(&self, session_id: SessionId) -> u16 {
        let mut bindings = self.bindings.write().unwrap_or_else(|e| e.into_inner());
        if let Some(b) = bindings.values().find(|b| b.session_id == session_id) {
            return b.port;
        }
        let port = if self.session_port_base == 0 {
            0
        } else {
            self.next_port.fetch_add(1, Ordering::SeqCst)
        };
        bindings.insert(
            port,
            SessionBinding {
                session_id,
                port,
                bound_at_epoch_ms: now_epoch_ms(),
            },
        );
        port
    }

    /// 占位端口 → 实际端口回填（单锁内完成，防竞态；仅 base=0 时产生差异）。
    pub fn rebind_port(&self, placeholder: u16, actual: u16) {
        if placeholder == actual {
            return;
        }
        let mut bindings = self.bindings.write().unwrap_or_else(|e| e.into_inner());
        if let Some(mut b) = bindings.remove(&placeholder) {
            b.port = actual;
            bindings.insert(actual, b);
        }
    }

    /// 回收端口（会话 Stable 超时后调用，§4.2）。
    pub fn release_port(&self, port: u16) -> bool {
        self.served
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&port);
        self.bindings
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&port)
            .is_some()
    }

    fn mark_served(&self, port: u16) {
        self.served
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(port);
    }

    fn is_served(&self, port: u16) -> bool {
        self.served
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&port)
    }

    pub fn binding_of_port(&self, port: u16) -> Option<SessionBinding> {
        self.bindings
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&port)
            .cloned()
    }

    pub fn session_of_port(&self, port: u16) -> Option<SessionId> {
        self.binding_of_port(port).map(|b| b.session_id)
    }

    pub fn list_bindings(&self) -> Vec<SessionBinding> {
        let mut v: Vec<SessionBinding> = self
            .bindings
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .cloned()
            .collect();
        v.sort_by_key(|b| b.port);
        v
    }

    pub fn list_unattributed(&self) -> Vec<UnattributedEntry> {
        self.unattributed
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// 注入验证上下文（启动期一次；重复注入时首个生效）。
    pub fn set_verify_context(&self, ctx: VerifyContext) {
        let _ = self.verify.set(ctx);
    }

    pub fn verify_context(&self) -> VerifyContext {
        self.verify.get().cloned().unwrap_or_default()
    }

    /// 注入原文旁挂目录（启动期一次；Some = `--store-content` 启用）。
    pub fn set_content_dir(&self, dir: Option<std::path::PathBuf>) {
        let _ = self.content_dir.set(dir);
    }

    pub fn content_dir(&self) -> Option<std::path::PathBuf> {
        self.content_dir.get().cloned().flatten()
    }
}

/// 请求路径/体 → provider 识别。
pub fn detect_provider(path: &str, body: Option<&serde_json::Value>) -> ProviderKind {
    if path.contains("/messages") {
        return ProviderKind::Anthropic;
    }
    if path.contains("/api/chat") || path.contains("/api/generate") {
        return ProviderKind::Ollama;
    }
    let _ = body;
    ProviderKind::OpenAi
}

fn provider_impl(kind: ProviderKind) -> &'static dyn LlmProvider {
    match kind {
        ProviderKind::Anthropic => &AnthropicProvider,
        ProviderKind::DeepSeek => &DeepSeekProvider,
        ProviderKind::Ollama => &OllamaProvider,
        ProviderKind::OpenAi | ProviderKind::Custom => &OpenAiProvider,
    }
}

fn upstream_base(up: &Upstreams, kind: ProviderKind) -> &str {
    match kind {
        ProviderKind::Anthropic => &up.anthropic,
        ProviderKind::DeepSeek => &up.deepseek,
        ProviderKind::Ollama => &up.ollama,
        ProviderKind::OpenAi | ProviderKind::Custom => &up.openai,
    }
}

/// 逐跳头（不透传，RFC 7230；host/content-length 由客户端重算）。
fn is_hop_by_hop(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
            | "host"
            | "content-length"
    )
}

fn forward_headers(headers: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (k, v) in headers {
        if !is_hop_by_hop(k.as_str()) {
            out.append(k, v.clone());
        }
    }
    out
}

/// 快照侧结果：Captured=已入链（旁挂成功或未启用）；Passthrough=体不可解析
/// 等纯透传不观测；Failed=入链/旁挂失败（fail-fast，调用方回 500 可见，R4）。
enum SnapshotOutcome {
    Captured(uuid::Uuid),
    Passthrough,
    Failed(String),
}

/// 快照侧：请求体 → 骨架/组成 → 入链（+`--store-content` 时原文旁挂）。
fn take_snapshot(
    state: &ProxyState,
    session: Option<(SessionId, BindingKind)>,
    kind: ProviderKind,
    body: &serde_json::Value,
) -> SnapshotOutcome {
    let p = provider_impl(kind);
    let Ok(extracted) = p.extract_request(body) else {
        return SnapshotOutcome::Passthrough;
    };
    let Ok(counts) = p.count_tokens(&extracted) else {
        return SnapshotOutcome::Passthrough;
    };
    let window =
        crate::provider::window_of(kind, &extracted.model, extracted.request_options.as_ref());
    let tools = crate::provider::skeleton_tools(&extracted).unwrap_or_default();
    let skeleton = ContextSkeleton::new(
        kind,
        extracted.model.clone(),
        extracted.system.as_deref().unwrap_or_default().as_bytes(),
        tools,
        extracted
            .messages
            .iter()
            .map(|m| {
                MessageEntry::new(
                    m.role.clone(),
                    m.content_kind,
                    m.text.as_bytes(),
                    m.media_count,
                )
            })
            .collect(),
    );
    let composition = match Composition::new(
        counts.system_tokens,
        counts.tools_tokens,
        counts.messages_tokens,
        counts.images_tokens,
        u32::try_from(extracted.messages.len()).unwrap_or(u32::MAX),
        u32::try_from(extracted.tools.len()).unwrap_or(u32::MAX),
        crate::TokenSource::Estimated,
    ) {
        Ok(c) => c,
        Err(e) => return SnapshotOutcome::Failed(format!("组成核账失败: {e}")),
    };
    let (sid, binding) = match session {
        Some((sid, binding)) => (Some(sid), binding),
        None => (None, BindingKind::Unattributed),
    };
    let snapshot = match ContextSnapshot::new(
        uuid::Uuid::new_v4(),
        SessionLink::new(sid, binding, None),
        now_epoch_ms(),
        kind,
        extracted.model.clone(),
        window,
        composition,
        skeleton,
    ) {
        Ok(s) => s,
        Err(e) => return SnapshotOutcome::Failed(format!("快照构造失败: {e}")),
    };
    let id = snapshot.snapshot_id;
    let model = snapshot.model.clone();
    if let Err(e) = state.chain.append(InspectorRecord::Snapshot(snapshot)) {
        return SnapshotOutcome::Failed(format!("链追加失败: {e}"));
    }
    // 原文旁挂（--store-content 隐私例外，§1 决策 9）：写失败=承诺破裂，必须可见
    if let Some(dir) = state.content_dir() {
        if let Err(e) = store_content_original(dir, id, kind, &extracted) {
            return SnapshotOutcome::Failed(format!(
                "原文旁挂写入失败（--store-content 承诺落原文，不可静默）: {e}"
            ));
        }
    }
    if session.is_none() {
        state
            .unattributed
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push(UnattributedEntry {
                snapshot_id: id,
                taken_at_epoch_ms: now_epoch_ms(),
                model,
            });
    }
    SnapshotOutcome::Captured(id)
}

/// 原文旁挂：单条 JSONL（content.jsonl，snapshot_id 关联），单句柄写+sync。
fn store_content_original(
    dir: std::path::PathBuf,
    id: uuid::Uuid,
    kind: ProviderKind,
    extracted: &crate::provider::ExtractedRequest,
) -> std::io::Result<()> {
    use std::io::Write as _;
    let line = serde_json::json!({
        "snapshot_id": id,
        "taken_at_epoch_ms": now_epoch_ms(),
        "provider": kind.as_str(),
        "model": extracted.model,
        "system": extracted.system,
        "tools": extracted.tools.iter().map(|t| &t.definition).collect::<Vec<_>>(),
        "messages": extracted.messages.iter().map(|m| serde_json::json!({
            "role": m.role, "text": m.text, "media_count": m.media_count,
        })).collect::<Vec<_>>(),
    });
    let path = dir.join("content.jsonl");
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    writeln!(f, "{}", serde_json::to_string(&line).unwrap_or_default())?;
    f.flush()?;
    f.sync_all()
}

/// 观测侧：非流式响应 → usage/finish → 入链。
fn observe_response(state: &ProxyState, kind: ProviderKind, snapshot_id: uuid::Uuid, body: &[u8]) {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return; // 上游非 JSON（错误页等）→ 不虚构观测
    };
    let Some(ex) = provider_impl(kind).extract_response(&v).ok() else {
        return;
    };
    let truncated =
        crate::metrics::detect_truncation(kind, ex.finish_reason.as_deref(), None, None);
    let obs = ResponseObservation::new(
        uuid::Uuid::new_v4(),
        snapshot_id,
        now_epoch_ms(),
        ex.finish_reason,
        ex.usage,
        truncated,
        false,
    );
    let _ = state.chain.append(InspectorRecord::Observation(obs));
}

/// 同步切行：把 chunk 追加进缓冲，吐出全部完整行（含 data: 前缀原样）。
fn split_lines(buffer: &mut String, chunk: &[u8]) -> Vec<String> {
    buffer.push_str(&String::from_utf8_lossy(chunk));
    let mut lines = Vec::new();
    while let Some(pos) = buffer.find('\n') {
        let line: String = buffer.drain(..=pos).collect();
        let trimmed = line.trim_end_matches(['\n', '\r']).to_owned();
        if !trimmed.trim().is_empty() {
            lines.push(trimmed);
        }
    }
    lines
}

/// 单行 → 事件 → 聚合器（行内观测不落账；流尾统一汇总入链）。
fn observe_line(
    _state: &ProxyState,
    kind: ProviderKind,
    agg: &mut StreamAggregator,
    raw_line: &str,
) {
    let payload = raw_line
        .strip_prefix("data: ")
        .filter(|p| *p != "[DONE]")
        .map(ToString::to_string)
        .unwrap_or_else(|| raw_line.to_owned());
    let Ok(chunk) = serde_json::from_str::<serde_json::Value>(&payload) else {
        return; // SSE 注释/心跳等 → 容错跳过（Unrecognized 语义）
    };
    agg.observe(provider_impl(kind).extract_stream_event(&chunk));
}

/// 代理处理核心（session 口与 admin fallback 共用）。
/// `session = None` 表示 admin 口未识别流量 → unattributed 桶。
pub async fn proxy_pass(
    state: Arc<ProxyState>,
    session: Option<(SessionId, BindingKind)>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body_bytes: bytes::Bytes,
) -> Response {
    let path = uri.path().to_owned();
    let query = uri.query().map(|q| format!("?{q}")).unwrap_or_default();
    let mut body_value: Option<serde_json::Value> = if body_bytes.is_empty() {
        None
    } else {
        serde_json::from_slice(&body_bytes).ok()
    };
    let kind = detect_provider(&path, body_value.as_ref());

    // 流式 OpenAI 请求：透传前注入 stream_options（改写后才允许算骨架，§4.3）
    let wants_stream = body_value
        .as_ref()
        .and_then(|b| b.get("stream"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if wants_stream && kind == ProviderKind::OpenAi {
        if let Some(v) = body_value.as_mut() {
            inject_stream_options_include_usage(v);
        }
    }

    // 快照侧（请求发出前定格）
    let snapshot_id = match body_value
        .as_ref()
        .map(|v| take_snapshot(&state, session, kind, v))
    {
        Some(SnapshotOutcome::Captured(id)) => Some(id),
        // 入链/旁挂失败 → 500 可见（R4 fail-fast，不静默吞掉观测承诺）
        Some(SnapshotOutcome::Failed(e)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                serde_json::json!({"error": e}).to_string(),
            )
                .into_response();
        }
        _ => None,
    };

    // 转发
    let base = upstream_base(&state.upstreams, kind).trim_end_matches('/');
    let url = format!("{base}{path}{query}");
    let mut req = state
        .http
        .request(method.clone(), &url)
        .headers(forward_headers(&headers));
    if !body_bytes.is_empty() {
        let payload: bytes::Bytes = match &body_value {
            Some(v) => serde_json::to_vec(v)
                .unwrap_or_else(|_| body_bytes.to_vec())
                .into(),
            None => body_bytes.clone(),
        };
        req = req.body(payload);
    }
    let upstream = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                serde_json::json!({"error": format!("上游不可达: {e}")}).to_string(),
            )
                .into_response();
        }
    };

    let status = upstream.status();
    let resp_headers = forward_headers(upstream.headers());
    let is_sse = upstream
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/event-stream"));

    if !is_sse {
        let resp_body = upstream.bytes().await.unwrap_or_default();
        if let Some(sid) = snapshot_id {
            observe_response(&state, kind, sid, &resp_body);
        }
        let mut resp = Response::builder().status(status);
        for (k, v) in &resp_headers {
            resp = resp.header(k, v);
        }
        return resp
            .body(Body::from(resp_body))
            .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response());
    }

    sse_passthrough(state, kind, snapshot_id, upstream, status, resp_headers).await
}

/// SSE 零缓冲透传：转发任务独占聚合器（逐行喂入），mpsc 通道交给响应体；
/// 流结束后以 `stream:true` 落一条汇总观测（§4.3）。
async fn sse_passthrough(
    state: Arc<ProxyState>,
    kind: ProviderKind,
    snapshot_id: Option<uuid::Uuid>,
    upstream: reqwest::Response,
    status: StatusCode,
    resp_headers: HeaderMap,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(16);
    let state_task = state.clone();
    tokio::spawn(async move {
        let mut agg = StreamAggregator::new();
        let mut partial = String::new();
        let mut stream = upstream.bytes_stream();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    for line in split_lines(&mut partial, &bytes) {
                        observe_line(&state_task, kind, &mut agg, &line);
                    }
                    if tx.send(Ok(bytes)).await.is_err() {
                        break; // 下游断开
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                    break;
                }
            }
        }
        // 半行 flush（末 chunk 可能不带换行）
        if !partial.trim().is_empty() {
            observe_line(&state_task, kind, &mut agg, &partial);
        }
        if let Some(sid) = snapshot_id {
            let out = agg.finish();
            let finish = out.finish_reason.clone();
            let obs = ResponseObservation::new(
                uuid::Uuid::new_v4(),
                sid,
                now_epoch_ms(),
                finish.clone(),
                out.usage,
                crate::metrics::detect_truncation(kind, finish.as_deref(), None, None),
                true,
            );
            let _ = state_task.chain.append(InspectorRecord::Observation(obs));
        }
    });
    let mut resp = Response::builder().status(status);
    for (k, v) in &resp_headers {
        resp = resp.header(k, v);
    }
    resp.body(Body::from_stream(
        tokio_stream::wrappers::ReceiverStream::new(rx),
    ))
    .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

// ============================================================
// admin 路由（19100）：绑定管理 + 代理 fallback（+ api.rs 查询路由合并）
// ============================================================

#[derive(Deserialize)]
pub struct BindRequest {
    pub session_id: SessionId,
}

async fn admin_bind(
    State(state): State<Arc<ProxyState>>,
    axum::Json(req): axum::Json<BindRequest>,
) -> Response {
    // 绑定即监听：拉起逐会话口（幂等；base=0 时端口为 OS 分配）
    match serve_session(state.clone(), req.session_id).await {
        Ok(port) => (
            StatusCode::CREATED,
            serde_json::json!({"session_id": req.session_id, "port": port}).to_string(),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            serde_json::json!({"error": format!("逐会话监听拉起失败: {e}")}).to_string(),
        )
            .into_response(),
    }
}

async fn admin_unbind(
    State(state): State<Arc<ProxyState>>,
    AxumPath(port): AxumPath<u16>,
) -> Response {
    if state.release_port(port) {
        StatusCode::NO_CONTENT.into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

async fn admin_list_bindings(State(state): State<Arc<ProxyState>>) -> Response {
    serde_json::to_string(&state.list_bindings())
        .unwrap_or_else(|_| "[]".into())
        .into_response()
}

/// admin 口代理 fallback：无绑定流量 → unattributed 桶。
async fn admin_proxy_fallback(
    State(state): State<Arc<ProxyState>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: bytes::Bytes,
) -> Response {
    proxy_pass(state, None, method, uri, headers, body).await
}

/// 逐会话口路由：整口代理（归属 = 绑定会话）。
pub fn session_router(state: Arc<ProxyState>, session_id: SessionId) -> Router {
    let for_handler = state.clone();
    Router::new()
        .fallback(
            move |method: Method, uri: Uri, headers: HeaderMap, body: bytes::Bytes| {
                let state = for_handler.clone();
                async move {
                    proxy_pass(
                        state,
                        Some((session_id, BindingKind::PortBinding)),
                        method,
                        uri,
                        headers,
                        body,
                    )
                    .await
                }
            },
        )
        .with_state(state)
}

/// admin 口路由（19100）。`extra` 供 api.rs 合并查询路由（Phase 3 接线）。
pub fn admin_router(state: Arc<ProxyState>, extra: Router<Arc<ProxyState>>) -> Router {
    extra
        .route("/admin/bindings", post(admin_bind).get(admin_list_bindings))
        .route("/admin/bindings/{port}", delete(admin_unbind))
        .route("/health", get(|| async { "ok" }))
        .fallback(admin_proxy_fallback)
        .with_state(state)
}

/// 便捷入口：绑定会话并拉起逐会话监听（返回实际端口；base=0 时为 OS 分配）。
pub async fn serve_session(state: Arc<ProxyState>, session_id: SessionId) -> std::io::Result<u16> {
    let placeholder = state.bind_session(session_id);
    if state.is_served(placeholder) {
        return Ok(placeholder); // 幂等：监听器已在
    }
    let app = session_router(state.clone(), session_id);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", placeholder)).await?;
    let actual = listener.local_addr()?.port();
    state.rebind_port(placeholder, actual);
    state.mark_served(actual);
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok(actual)
}
