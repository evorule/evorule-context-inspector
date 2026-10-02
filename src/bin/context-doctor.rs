//! evorule-context-doctor —— 上下文诊断 CLI（C-6 一期零侵入路线）。
//!
//! 用法：
//! ```text
//! evorule-context-doctor --fact-wal <session fact WAL 路径>
//!                        [--wire-wal <inspector.log 路径>]
//!                        [--constraints <约束清单文件，每行一条>]
//!                        [--json]   机器可读输出（缺省文本）
//! ```
//!
//! 输出双格式（T-3 附加建议）；确定性自证：同输入两次运行逐字节一致（RL-A3）。

use evorule_llm_context_inspector::doctor::{
    attribute, extract_constraints_heuristic, extract_fact_face, extract_wire_face,
    load_constraints, run_checks,
};
use std::path::PathBuf;

struct CliArgs {
    fact_wal: PathBuf,
    wire_wal: Option<PathBuf>,
    constraints: Option<PathBuf>,
    json: bool,
}

fn parse_args() -> Result<CliArgs, String> {
    let mut args = CliArgs {
        fact_wal: PathBuf::new(),
        wire_wal: None,
        constraints: None,
        json: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--fact-wal" => args.fact_wal = PathBuf::from(it.next().ok_or("--fact-wal 缺值")?),
            "--wire-wal" => {
                args.wire_wal = Some(PathBuf::from(it.next().ok_or("--wire-wal 缺值")?))
            }
            "--constraints" => {
                args.constraints = Some(PathBuf::from(it.next().ok_or("--constraints 缺值")?))
            }
            "--json" => args.json = true,
            other => return Err(format!("未知参数 {other}")),
        }
    }
    if args.fact_wal.as_os_str().is_empty() {
        return Err("缺 --fact-wal（session fact WAL 路径）".into());
    }
    Ok(args)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("参数错误: {e}");
            std::process::exit(2);
        }
    };
    let report = match run(&args) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("诊断失败: {e}");
            std::process::exit(1);
        }
    };
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
        );
    } else {
        print_text(&report);
    }
}

fn run(args: &CliArgs) -> Result<evorule_llm_context_inspector::doctor::DiagnosisReport, String> {
    let (face, snaps) = extract_fact_face(&args.fact_wal).map_err(|e| e.to_string())?;
    let wire_face = match &args.wire_wal {
        Some(p) => Some(extract_wire_face(p).map_err(|e| e.to_string())?),
        None => None,
    };

    // 诊断对象 = fact 面最新一份 wire 快照（首轮主调用或最新 audited 调用）
    let Some(latest) = snaps.last().cloned() else {
        return Err("fact WAL 中没有携带 messages 的 Command wire 快照（无可诊断轮）".into());
    };
    let constraints = match &args.constraints {
        Some(p) => load_constraints(p).map_err(|e| e.to_string())?,
        None => {
            let system = latest
                .messages
                .iter()
                .find(|m| m.role == "system")
                .map(|m| m.content.clone())
                .unwrap_or_default();
            extract_constraints_heuristic(&system)
        }
    };

    let model = "deepseek-chat".to_owned(); // token 估算编码器选择（近似口径，ProviderReported 哲学同源）
    let attribution = attribute(&latest.messages, latest.tools, |text| {
        evorule_llm_context_inspector::tokens::estimate_text(&model, text).unwrap_or(0)
    });
    let checks = run_checks(&latest.messages, &attribution, &constraints);

    let blind_spot = wire_face
        .as_ref()
        .map(|w| w.observed_rounds.saturating_sub(face.wire_snapshots));

    Ok(evorule_llm_context_inspector::doctor::DiagnosisReport {
        fact_face: evorule_llm_context_inspector::doctor::FactFaceSummary {
            total_facts: face.total_facts,
            wire_snapshots: face.wire_snapshots,
            memory_originals: face.memory_originals,
            stable_entries: face.stable_entries,
            io_requests: face.io_requests,
        },
        wire_face: wire_face
            .as_ref()
            .map(|w| evorule_llm_context_inspector::doctor::WireFaceSummary {
                observed_rounds: w.observed_rounds,
            }),
        attribution: attribution
            .iter()
            .filter(|a| a.role != "tools")
            .map(|a| evorule_llm_context_inspector::doctor::LayerAttribution {
                layer: a.layer.as_str().to_owned(),
                messages: 1,
                tokens_est: a.tokens_est,
                share_permyriad: 0,
            })
            .collect(),
        checks,
        blind_spot_rounds: blind_spot,
        notes: vec![
            "wire 面为旁路观测（inspector MITM），非 RL-A2 记账权威；记账面收敛（回喂轮 wire 落链）为 evo-agent 侧独立议题".into(),
            "token 为 tiktoken 近似口径（ProviderReported 哲学同源）".into(),
        ],
    })
}

fn print_text(report: &evorule_llm_context_inspector::doctor::DiagnosisReport) {
    println!("=== 上下文诊断报告（零侵入，只读） ===");
    println!(
        "fact 面: facts={} wire快照={} 记忆原文={} stable={} io={}",
        report.fact_face.total_facts,
        report.fact_face.wire_snapshots,
        report.fact_face.memory_originals,
        report.fact_face.stable_entries,
        report.fact_face.io_requests
    );
    if let Some(w) = &report.wire_face {
        println!("wire 面: 观测轮次={}", w.observed_rounds);
    }
    if let Some(bs) = report.blind_spot_rounds {
        println!("G-6 盲区: {bs} 轮（wire 观测 − fact 可重建；旁路观测不替代记账）");
    }
    println!("--- 八层归因（最新 wire） ---");
    for a in &report.attribution {
        println!("  {:<22} tokens≈{}", a.layer, a.tokens_est);
    }
    println!("--- 检查发现（{} 条） ---", report.checks.len());
    for c in &report.checks {
        let sev = match c.severity {
            evorule_llm_context_inspector::doctor::Severity::Info => "INFO",
            evorule_llm_context_inspector::doctor::Severity::Warning => "WARN",
            evorule_llm_context_inspector::doctor::Severity::Critical => "CRIT",
        };
        println!("  [{sev}] {}: {}", c.checker, c.message);
    }
    for n in &report.notes {
        println!("(注) {n}");
    }
}
