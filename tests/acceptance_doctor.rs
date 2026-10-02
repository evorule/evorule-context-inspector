//! 诊断工具一期验收（C-6 一期零侵入路线：04 §5.2 组成 + 06 §3.5 修正口径）。
//!
//! 覆盖：fact 面抽取（真实格式）/ 八层启发式归因 / 检查器（归因分布、工具膨胀、
//! 重复、约束生存率、I1 字面级）/ G-6 盲区量化 / 确定性自证（双跑逐字节一致）。
//! 全部输入为合成 WAL（单测口径）；真实 WAL 抽查由运维侧执行（07 档记录）。
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use evorule_llm_context_inspector::chain::{InspectorLog, InspectorRecord};
use evorule_llm_context_inspector::doctor::{
    attribute, extract_constraints_heuristic, extract_fact_face, extract_wire_face, run_checks,
};
use evorule_reactor::{Fact, FactId};
use std::sync::Arc;

/// 构造 tier1 哈希格式 fact WAL：1 条 Command（携带 messages wire）+ 1 条 PayloadUpdate(__memory__)。
fn build_fact_wal(path: &std::path::Path, wire_messages: serde_json::Value) {
    use std::io::Write as _;
    let instruction = serde_json::json!({
        "__meta__": {"caller_role": "llm"},
        "params": {
            "audit_purpose": "main",
            "messages": wire_messages,
            "tools": [{"type": "function", "function": {"name": "calc"}}]
        }
    });
    let tcb_instruction = evorule_reactor::serde_to_tcb(&instruction);
    let fact = Fact::Command {
        id: FactId(100),
        instruction: tcb_instruction,
    };
    let content_hash = evorule_reactor::fact_hash(&fact).unwrap();
    let chain_hash = evorule_reactor::chain_step("genesis", &content_hash);
    let fact_json = serde_json::json!({
        "type": "Command",
        "id": 100,
        "instruction": instruction,
    });
    let line = serde_json::json!({
        "version_before": 0,
        "fact": fact_json,
        "content_hash": content_hash,
        "prev_hash": "genesis",
        "chain_hash": chain_hash,
    });
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "{line}").unwrap();
    f.sync_all().unwrap();
}

fn wire_fixture() -> serde_json::Value {
    serde_json::json!([
        {"role": "system", "content": "你是财务助手。禁止编造数字。## Stable Facts\n- revenue=100"},
        {"role": "user", "content": "帮我核对季度收入"},
        {"role": "assistant", "content": "好的，开始核对。"},
        {"role": "tool", "content": "{\"rows\": 5000, \"raw\": \"...\"}"}
    ])
}

fn temp(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("doctor_{}_{}", tag, uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

// ---- fact 面抽取（第一锚点：落链格式实测）----

#[test]
fn doctor_fact_face_extracts_wire_and_counts() {
    let dir = temp("extract");
    let wal = dir.join("fact.log");
    build_fact_wal(&wal, wire_fixture());

    let (face, snaps) = extract_fact_face(&wal).unwrap_or_else(|e| panic!("抽取失败: {e}"));
    assert_eq!(face.total_facts, 1);
    assert_eq!(
        face.wire_snapshots, 1,
        "Command(params.messages) 计一条 wire"
    );
    assert_eq!(snaps.len(), 1);
    assert_eq!(snaps[0].purpose, "main");
    assert_eq!(snaps[0].messages.len(), 4);
    assert_eq!(snaps[0].tools, 1);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- 八层启发式归因 ----

#[test]
fn attribution_maps_layers_by_format_markers() {
    let messages: Vec<evorule_llm_context_inspector::doctor::FaceMessage> =
        serde_json::from_value(wire_fixture()).unwrap();
    let attribution = attribute(&messages, 1, |t| t.chars().count() as u64);

    // 分区标记优先：system 同时含主体与 ## Stable Facts 分区 → 启发式整条归 L6
    //（已知局限：单条 system 内"主体 vs 分区"不细分，报告按消息级归因如实呈现）
    assert_eq!(
        attribution[0].layer,
        evorule_llm_context_inspector::doctor::Layer::Knowledge
    );
    assert_eq!(
        attribution[1].layer,
        evorule_llm_context_inspector::doctor::Layer::Goal
    );
    assert_eq!(
        attribution[2].layer,
        evorule_llm_context_inspector::doctor::Layer::History
    );
    assert_eq!(
        attribution[3].layer,
        evorule_llm_context_inspector::doctor::Layer::Observation
    );
}

// ---- 检查器：约束生存率（shadow 策略 #1）+ I1 字面级 ----

#[test]
fn constraint_survival_and_literal_i1() {
    let messages: Vec<evorule_llm_context_inspector::doctor::FaceMessage> =
        serde_json::from_value(serde_json::json!([
            {"role": "system", "content": "禁止编造数字。必须引用来源。"},
            {"role": "user", "content": "核对"},
            {"role": "assistant", "content": "数字大概是 42，可以编造数字，也可以不引用来源。"}
        ]))
        .unwrap();
    let attribution = attribute(&messages, 0, |t| t.chars().count() as u64);
    let constraints = extract_constraints_heuristic("禁止编造数字。必须引用来源。");
    assert_eq!(constraints.source, "heuristic_system_prompt");
    assert_eq!(constraints.phrases.len(), 2);

    let checks = run_checks(&messages, &attribution, &constraints, &[]);
    let survival: Vec<_> = checks
        .iter()
        .filter(|c| c.checker == "constraint_presence")
        .collect();
    assert!(!survival.is_empty());
    let joined = survival
        .iter()
        .map(|c| c.message.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        joined.contains("0/2") || joined.contains("缺失"),
        "两条约束均被对冲，生存率应为 0：{joined}"
    );
    let i1: Vec<_> = checks
        .iter()
        .filter(|c| c.checker == "i1_authority_literal")
        .collect();
    assert!(
        !i1.is_empty(),
        "字面级对冲（可以编造数字）必须被 I1 检查器捕获"
    );
}

// ---- G-6 盲区量化（wire 面 + fact 面对照）----

#[tokio::test]
async fn blind_spot_quantified_when_wire_face_present() {
    let dir = temp("blindspot");
    let fact_wal = dir.join("fact.log");
    build_fact_wal(&fact_wal, wire_fixture());

    // wire 面：3 轮观测（> fact 面可重建的 1 份）
    let inspector_wal = dir.join("inspector.log");
    {
        let chain = Arc::new(InspectorLog::open(&inspector_wal).unwrap());
        for _ in 0..3 {
            chain
                .append(InspectorRecord::Snapshot(test_snapshot()))
                .unwrap();
        }
        drop(chain);
    }
    let wire = extract_wire_face(&inspector_wal).unwrap();
    assert_eq!(wire.observed_rounds, 3);
    let (face, _) = extract_fact_face(&fact_wal).unwrap();
    let blind = wire.observed_rounds.saturating_sub(face.wire_snapshots);
    assert_eq!(blind, 2, "G-6 盲区 = wire 观测 − fact 可重建");

    let _ = std::fs::remove_dir_all(&dir);
}

fn test_snapshot() -> evorule_llm_context_inspector::model::ContextSnapshot {
    use evorule_llm_context_inspector::model::{
        Composition, ContextSkeleton, ContextSnapshot, MessageEntry, SessionLink, ToolEntry,
    };
    use evorule_llm_context_inspector::{BindingKind, ContentKind, ProviderKind, TokenSource};
    let skeleton = ContextSkeleton::new(
        ProviderKind::OpenAi,
        "gpt-4o",
        b"sys",
        vec![ToolEntry::new("t", b"{}")],
        vec![MessageEntry::new("user", ContentKind::Text, b"m", 0)],
    );
    let composition = Composition::new(3, 2, 7, 0, 1, 1, TokenSource::Estimated).unwrap();
    ContextSnapshot::new(
        uuid::Uuid::new_v4(),
        SessionLink::new(None, BindingKind::Unattributed, None),
        0,
        ProviderKind::OpenAi,
        "gpt-4o",
        None,
        composition,
        skeleton,
    )
    .unwrap()
}

// ---- 确定性自证 + CLI 真跑（双跑逐字节一致）----

#[test]
fn cli_determinism_two_runs_byte_identical() {
    let dir = temp("cli");
    let fact_wal = dir.join("fact.log");
    build_fact_wal(&fact_wal, wire_fixture());

    let exe = env!("CARGO_BIN_EXE_evorule-context-doctor");
    let mut outputs = Vec::new();
    for _ in 0..2 {
        let out = std::process::Command::new(exe)
            .arg("--fact-wal")
            .arg(&fact_wal)
            .arg("--json")
            .output()
            .expect("CLI 运行失败");
        assert!(
            out.status.success(),
            "CLI 退出码非 0: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        outputs.push(out.stdout);
    }
    assert_eq!(
        outputs[0], outputs[1],
        "确定性自证：同输入两次运行必须逐字节一致"
    );
    let report: serde_json::Value = serde_json::from_slice(&outputs[0]).unwrap();
    assert_eq!(report["fact_face"]["wire_snapshots"], 1);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- I5 摘要保真（R3 方案 B 落地后可验证）----

#[tokio::test]
async fn i5_summary_fidelity_detects_lost_constraints() {
    use evorule_llm_context_inspector::doctor::ConstraintSource;
    use evorule_llm_context_inspector::doctor::{i5_summary_fidelity, RollingSummaryLanding};

    let constraints = ConstraintSource {
        source: "heuristic_system_prompt",
        phrases: vec!["禁止编造数字".to_owned(), "必须引用来源".to_owned()],
    };

    // 场景 1：摘要完整保留约束 → 仅 Info（无 Critical）
    let full = vec![RollingSummaryLanding {
        gen: 1,
        frozen_len_before: 0,
        frozen_len_after: 7,
        summary_text: "禁止编造数字，必须引用来源；已完成核对。".to_owned(),
    }];
    let r1 = i5_summary_fidelity(&full, &constraints);
    assert!(!r1
        .iter()
        .any(|c| c.severity == evorule_llm_context_inspector::doctor::Severity::Critical));

    // 场景 2：摘要丢「必须引用来源」→ Critical（F2 压缩丢失，对照 dropped 原文核验）
    let lossy = vec![RollingSummaryLanding {
        gen: 2,
        frozen_len_before: 7,
        frozen_len_after: 12,
        summary_text: "禁止编造数字的行为已记录。".to_owned(),
    }];
    let r2 = i5_summary_fidelity(&lossy, &constraints);
    let criticals: Vec<_> = r2
        .iter()
        .filter(|c| c.severity == evorule_llm_context_inspector::doctor::Severity::Critical)
        .collect();
    assert_eq!(criticals.len(), 1, "丢「必须引用来源」应报一条 Critical");
    assert!(criticals[0].message.contains("必须引用来源"));
}
