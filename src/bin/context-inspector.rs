//! evorule-context-inspector 插件进程入口（manifest 型，ai-plugin 同款形态）。
//!
//! 路由布局：
//! - admin 口（默认 127.0.0.1:19100）：`/health`、`/admin/bindings`
//!   （POST 绑定即拉起逐会话监听 / GET 列表 / DELETE 回收）、查询面
//!   （`/api/context/*`、`/api/sessions/{id}/context/*`）+ 代理 fallback
//!   （无绑定流量 → unattributed 桶）。
//! - 逐会话口（`--session-port-base` 起，默认 19101）：整口代理，归属绑定会话。
//!
//! 落盘（N 族链路 A 显式参数）：`--inspector-log-dir <绝对路径>` 指向插件
//! 自持 data 目录（WAL=inspector.log；`--store-content` 时原文旁挂
//! content-*.jsonl）；**未传 = 仅内存模式，不持久化**（合法形态）。
//! **loopback-only**：全部监听 127.0.0.1，部署文档须写明不可对外（J 族）。
//!
//! Header 纪律：Authorization 等只透传，永不入日志（§2.5）。

use evorule_llm_context_inspector::api::query_router;
use evorule_llm_context_inspector::chain::InspectorLog;
use evorule_llm_context_inspector::proxy::{admin_router, serve_session, ProxyState, Upstreams};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone)]
struct Args {
    admin_addr: String,
    session_port_base: u16,
    inspector_log_dir: Option<PathBuf>,
    store_content: bool,
    upstreams: Upstreams,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        admin_addr: "127.0.0.1:19100".into(),
        session_port_base: 19101,
        inspector_log_dir: None,
        store_content: false,
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
    let _ = args.store_content; // 原文旁挂在 Phase 3 存储增量接线（本轮链与代理优先）

    let state = Arc::new(ProxyState::new(
        chain,
        args.upstreams.clone(),
        args.session_port_base,
    ));
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
