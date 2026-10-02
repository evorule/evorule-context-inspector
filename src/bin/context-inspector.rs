//! evorule-context-inspector 插件进程入口（manifest 型，ai-plugin 同款形态）。
//!
//! 路由布局：
//! - admin 口（默认 127.0.0.1:19100）：`/health`、`/admin/bindings`
//!   （POST 绑定即拉起逐会话监听 / GET 列表 / DELETE 回收）、查询面
//!   （`/api/context/*`、`/api/sessions/{id}/context/*`）、契约服务面
//!   （`/services/context_*`）+ 代理 fallback（无绑定流量 → unattributed 桶）。
//! - 逐会话口（`--session-port-base` 起，默认 19101）：整口代理，归属绑定会话。
//!
//! 落盘（N 族链路 A 显式参数）：`--inspector-log-dir <绝对路径>` 指向插件
//! 自持 data 目录（WAL=inspector.log；锚点流=anchors.jsonl + anchor-key.json；
//! `--store-content` 时原文旁挂 content-*.jsonl）；**未传 = 仅内存模式，
//! 不持久化**（合法形态）。**loopback-only**：全部监听 127.0.0.1，
//! 部署文档须写明不可对外（J 族）。
//!
//! Header 纪律：Authorization 等只透传，永不入日志（§2.5）。

use evorule_llm_context_inspector::anchor::AnchorService;
use evorule_llm_context_inspector::api::query_router;
use evorule_llm_context_inspector::chain::InspectorLog;
use evorule_llm_context_inspector::proxy::{
    admin_router, serve_session, ProxyState, Upstreams, VerifyContext,
};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone)]
struct Args {
    admin_addr: String,
    session_port_base: u16,
    inspector_log_dir: Option<PathBuf>,
    store_content: bool,
    fact_log: Option<PathBuf>,
    anchor_every_n: u64,
    upstreams: Upstreams,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        admin_addr: "127.0.0.1:19100".into(),
        session_port_base: 19101,
        inspector_log_dir: None,
        store_content: false,
        fact_log: None,
        anchor_every_n: evorule_llm_context_inspector::anchor::ANCHOR_EVERY_N_DEFAULT,
        upstreams: Upstreams::default(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--admin-addr" => {
                args.admin_addr = it.next().ok_or("--admin-addr 缺值")?;
            }
            "--session-port-base" => {
                let v = it.next().ok_or("--session-port-base 缺值")?;
                args.session_port_base = v
                    .parse()
                    .map_err(|_| format!("--session-port-base 非法: {v}"))?;
            }
            "--inspector-log-dir" => {
                let v = it.next().ok_or("--inspector-log-dir 缺值")?;
                args.inspector_log_dir = Some(PathBuf::from(v));
            }
            "--store-content" => args.store_content = true,
            "--fact-log" => {
                args.fact_log = Some(PathBuf::from(it.next().ok_or("--fact-log 缺值")?));
            }
            "--anchor-every-n" => {
                let v = it.next().ok_or("--anchor-every-n 缺值")?;
                args.anchor_every_n = v
                    .parse()
                    .map_err(|_| format!("--anchor-every-n 非法: {v}"))?;
            }
            "--upstream-openai" => {
                args.upstreams.openai = it.next().ok_or("--upstream-openai 缺值")?
            }
            "--upstream-anthropic" => {
                args.upstreams.anthropic = it.next().ok_or("--upstream-anthropic 缺值")?
            }
            "--upstream-deepseek" => {
                args.upstreams.deepseek = it.next().ok_or("--upstream-deepseek 缺值")?
            }
            "--upstream-ollama" => {
                args.upstreams.ollama = it.next().ok_or("--upstream-ollama 缺值")?
            }
            other => return Err(format!("未知参数 {other}（--help 查看用法）")),
        }
    }
    Ok(args)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    let chain = match &args.inspector_log_dir {
        Some(dir) => {
            std::fs::create_dir_all(dir)?;
            let wal = dir.join("inspector.log");
            let log = InspectorLog::open(&wal)?;
            eprintln!(
                "[context-inspector] WAL 已打开: {}（已有 {} 条记录）",
                wal.display(),
                log.total_records()?
            );
            Arc::new(log)
        }
        None => {
            eprintln!("[context-inspector] 未传 --inspector-log-dir：仅内存模式，不持久化");
            Arc::new(InspectorLog::in_memory())
        }
    };
    if args.store_content && args.inspector_log_dir.is_none() {
        return Err(
            "--store-content 需要 --inspector-log-dir（原文旁挂必须落盘，拒绝静默丢弃）".into(),
        );
    }

    let state = Arc::new(ProxyState::new(
        chain.clone(),
        args.upstreams.clone(),
        args.session_port_base,
    ));
    if let Some(dir) = args
        .store_content
        .then(|| args.inspector_log_dir.clone())
        .flatten()
    {
        state.set_content_dir(Some(dir.clone()));
        eprintln!(
            "[context-inspector] --store-content 已启用：原文旁挂 {}/content.jsonl（隐私例外模式）",
            dir.display()
        );
    }
    // 验证上下文：锚点服务（data 目录就绪时）+ 对账路径（N 族链路 A 显式参数）
    if let Some(dir) = &args.inspector_log_dir {
        let anchors = AnchorService::load_or_create(dir, chain.clone(), args.anchor_every_n)?;
        eprintln!(
            "[context-inspector] 锚点流: {}（每 {} 条一锚，Ed25519 自持，格式复用 cli anchor_payload）",
            anchors.anchors_path().display(),
            args.anchor_every_n
        );
        state.set_verify_context(VerifyContext {
            anchors: Some(Arc::new(anchors)),
            inspector_wal_path: Some(dir.join("inspector.log")),
            fact_log_path: args.fact_log.clone(),
        });
    }

    let app = admin_router(state, query_router());
    let listener = tokio::net::TcpListener::bind(&args.admin_addr).await?;
    eprintln!(
        "[context-inspector] admin 口监听 {}（loopback-only，不可对外）；逐会话口基址 {}",
        args.admin_addr, args.session_port_base
    );
    axum::serve(listener, app).await?;
    Ok(())
}

/// 绑定即监听：admin_bind 成功后由调用方（proxy 层）经此拉起逐会话口。
/// 供集成测试与后续 admin 流程复用。
pub async fn bind_and_serve(state: Arc<ProxyState>, session_id: u64) -> std::io::Result<u16> {
    serve_session(state, session_id).await
}
