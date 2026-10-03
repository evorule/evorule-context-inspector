//! 上下文诊断工具一期（C-6 裁定：零侵入链上读取器，方案 A）。
//!
//! 双面观测（docs/dual-face-observation.md）：
//! - **fact 面**：session fact WAL → Command(params.messages=audited 调用 wire)、
//!   PayloadUpdate(原始全文)、IoRequest、Stable——事实面全量 + 首轮 wire；
//! - **wire 面**（可选输入）：inspector WAL（MITM 旁路）——全部轮次实收；
//! - **对照**：wire 轮次 vs fact 可重建轮次 → G-6 盲区量化（不虚构不掩盖）。
//!
//! 确定性红线（RL-A3）：本模块全部为纯函数——同输入必得同报告；
//! 禁时间/随机/外部调用。诊断只读，不写事实层（RL-A1）。

use crate::chain::{InspectorRecord, WalLine};
use std::collections::{BTreeMap, HashSet};
use std::io::BufRead;
use std::path::Path;

/// 参考模型八层（04 档 §11.2 适配器目标层）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
pub enum Layer {
    /// L1-L3 身份/约束/指令——wire 首条 system 的非分区主体（启发式不可再分，合并桶）
    SystemCore,
    /// L4 目标——首轮 user（goal）
    Goal,
    /// L5 能力——tools 定义
    Capability,
    /// L6 知识——记忆召回分区（## Stable Facts / ## Previous Sessions / ## Relevant Events）
    Knowledge,
    /// L7 观察——tool 角色回喂
    Observation,
    /// L8 历史——assistant/user 后续轮 + hint/摘要
    History,
}

impl Layer {
    pub fn as_str(self) -> &'static str {
        match self {
            Layer::SystemCore => "L1-L3_system_core",
            Layer::Goal => "L4_goal",
            Layer::Capability => "L5_capability",
            Layer::Knowledge => "L6_knowledge",
            Layer::Observation => "L7_observation",
            Layer::History => "L8_history",
        }
    }
}

/// 一条 wire 消息的归因结果。
#[derive(Debug, Clone, serde::Serialize)]
pub struct AttributedMessage {
    pub layer: Layer,
    pub role: String,
    pub index: usize,
    pub chars: usize,
    pub tokens_est: u64,
    /// 分区标记（L6 内的 ## 段名；非分区为 None）
    pub section: Option<String>,
}

/// fact 面抽取结果。
/// 一条滚动摘要落链记录（R3 方案 B：PayloadUpdate rolling_summary）。
#[derive(Debug, Clone, serde::Serialize)]
pub struct RollingSummaryLanding {
    pub gen: u64,
    pub frozen_len_before: u64,
    pub frozen_len_after: u64,
    pub summary_text: String,
}

#[derive(Debug, Default, Clone)]
pub struct FactFace {
    /// Command facts 中携带的 LLM 调用 wire 条数
    pub wire_snapshots: u64,
    /// 滚动摘要落链记录（R3 方案 B；I5 保真检查输入）
    pub rolling_summaries: Vec<RollingSummaryLanding>,
    /// PayloadUpdate 原始全文条数（__memory__）
    pub memory_originals: u64,
    /// Stable（sediment 摘要/稳定事实/事件）条数
    pub stable_entries: u64,
    /// IoRequest 条数
    pub io_requests: u64,
    /// IoResponse 带真值 usage 的记录（provider 真值 token 对照）
    pub usage_trues: Vec<UsageTruth>,
    pub total_facts: u64,
}

/// F-901 真值校准：provider 回报的真实 usage（对照 tiktoken 估算）。
#[derive(Debug, Clone, serde::Serialize)]
pub struct UsageTruth {
    /// prompt_tokens（provider 真值）
    pub prompt_tokens: u64,
    /// completion_tokens（provider 真值）
    pub completion_tokens: u64,
    /// tiktoken 对同内容的估算值（None = 无对应 wire 快照）
    pub tokens_est: Option<u64>,
}

/// 一份 LLM 调用 wire（来自 fact 面）。
#[derive(Debug, Clone, serde::Serialize)]
pub struct WireSnapshot {
    pub fact_id: u64,
    /// 调用目的（audit_purpose；无则 "unknown"）
    pub purpose: String,
    pub messages: Vec<FaceMessage>,
    pub tools: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FaceMessage {
    pub role: String,
    pub content: String,
}

/// wire 面抽取结果（inspector WAL，可选）。
#[derive(Debug, Default, Clone)]
pub struct WireFace {
    /// MITM 观测到的 LLM 调用轮次（每条 snapshot 记录 = 一次调用）
    pub observed_rounds: u64,
}

/// 归因 + 检查报告（JSON/文本双输出的数据体）。
#[derive(Debug, Clone, serde::Serialize)]
pub struct DiagnosisReport {
    pub fact_face: FactFaceSummary,
    pub wire_face: Option<WireFaceSummary>,
    /// wire 面最新一轮的八层 token 归因（wire 面缺省时取 fact 面最新 wire 快照）
    pub attribution: Vec<LayerAttribution>,
    pub checks: Vec<CheckFinding>,
    /// G-6 盲区量化：wire 观测轮次 - fact 可重建轮次（两 face 齐备时才有值）
    pub blind_spot_rounds: Option<u64>,
    /// 时间窗/轮次能力声明（如实降级口径，与 reconcile 一致）
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerAttribution {
    pub layer: String,
    pub messages: u64,
    pub tokens_est: u64,
    pub share_permyriad: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FactFaceSummary {
    pub total_facts: u64,
    pub wire_snapshots: u64,
    pub memory_originals: u64,
    pub stable_entries: u64,
    pub io_requests: u64,
    /// 滚动摘要落链条数（R3 方案 B）
    pub rolling_summaries: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WireFaceSummary {
    pub observed_rounds: u64,
}

/// 单条检查发现（可行动口径：指向哪条消息/哪个面/建议）。
#[derive(Debug, Clone, serde::Serialize)]
pub struct CheckFinding {
    pub checker: &'static str,
    pub severity: Severity,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

/// 约束清单来源（单一权威源；文件优先，缺省启发提取自首轮 system）。
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConstraintSource {
    pub source: &'static str,
    pub phrases: Vec<String>,
}

/// 从约束清单文件装载（每行一条）。
pub fn load_constraints(path: &Path) -> Result<ConstraintSource, crate::InspectorError> {
    let raw =
        std::fs::read_to_string(path).map_err(|e| crate::InspectorError::Io(e.to_string()))?;
    let phrases: Vec<String> = raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect();
    Ok(ConstraintSource {
        source: "file",
        phrases,
    })
}

/// 启发提取（缺省：首轮 system 中含强约束模态词的句子）。
pub fn extract_constraints_heuristic(system: &str) -> ConstraintSource {
    let modals = [
        "必须",
        "禁止",
        "不得",
        "不能",
        "永远",
        "一定要",
        "must",
        "never",
        "always",
    ];
    let phrases: Vec<String> = system
        .split(['。', '\n', '；', ';'])
        .filter(|s| modals.iter().any(|m| s.to_lowercase().contains(m)))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    ConstraintSource {
        source: "heuristic_system_prompt",
        phrases,
    }
}

// ---------- fact 面抽取 ----------

/// 从 IoResponse 的 result JSON 中提取 provider usage 真值（F-901）。
fn extract_usage_truth(v: &serde_json::Value) -> Option<UsageTruth> {
    let usage = v.get("usage")?;
    let pt = usage.get("prompt_tokens")?.as_u64()?;
    let ct = usage.get("completion_tokens")?.as_u64()?;
    Some(UsageTruth {
        prompt_tokens: pt,
        completion_tokens: ct,
        tokens_est: None,
    })
}

/// 从 session fact WAL 抽取事实面（只读；解析失败行计数跳过，报告如实）。
pub fn extract_fact_face(
    wal: &Path,
) -> Result<(FactFace, Vec<WireSnapshot>), crate::InspectorError> {
    use evorule_reactor::read_wal_with_hash;
    let records = read_wal_with_hash(wal)
        .map_err(|e| crate::InspectorError::Io(format!("fact WAL 读取失败 {wal:?}: {e}")))?;
    let mut face = FactFace::default();
    let mut snaps: Vec<WireSnapshot> = Vec::new();
    face.total_facts = records.len() as u64;
    for r in &records {
        match &r.fact {
            evorule_reactor::Fact::Command { instruction, .. } => {
                let v = evorule_reactor::tcb_to_serde(instruction);
                // 实测落链形态（06 档第一锚点）：messages 在 instruction.params.messages
                //（audited 调用）；兼容顶层 messages 形态
                let body = v
                    .get("params")
                    .filter(|p| p.get("messages").is_some())
                    .unwrap_or(&v);
                if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
                    let purpose = body
                        .get("audit_purpose")
                        .and_then(|p| p.as_str())
                        .unwrap_or("unknown")
                        .to_owned();
                    let messages: Vec<FaceMessage> = msgs
                        .iter()
                        .map(|m| FaceMessage {
                            role: m
                                .get("role")
                                .and_then(|r| r.as_str())
                                .unwrap_or("")
                                .to_owned(),
                            content: match m.get("content") {
                                Some(serde_json::Value::String(s)) => s.clone(),
                                Some(other) => other.to_string(),
                                None => String::new(),
                            },
                        })
                        .collect();
                    let tools = body
                        .get("tools")
                        .and_then(|t| t.as_array())
                        .map(Vec::len)
                        .unwrap_or(0);
                    snaps.push(WireSnapshot {
                        fact_id: r.fact.id().0,
                        purpose,
                        messages,
                        tools,
                    });
                }
            }
            evorule_reactor::Fact::PayloadUpdate { path, value, .. } => {
                // value 为 TCB JsonValue——转 serde 后走统一访问器
                let v = evorule_reactor::tcb_to_serde(value);
                if path.ends_with("rolling_summary") {
                    // R3 方案 B 落链：摘要代账（I5 保真检查输入）
                    let gen = v.get("gen").and_then(|x| x.as_u64()).unwrap_or(0);
                    let before = v
                        .get("frozen_len_before")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(0);
                    let after = v
                        .get("frozen_len_after")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(0);
                    let text = v
                        .get("summary_text")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_owned();
                    face.rolling_summaries.push(RollingSummaryLanding {
                        gen,
                        frozen_len_before: before,
                        frozen_len_after: after,
                        summary_text: text,
                    });
                }
                if path.starts_with("__memory__") {
                    face.memory_originals += 1;
                }
            }
            evorule_reactor::Fact::IoRequest { .. } => face.io_requests += 1,
            evorule_reactor::Fact::IoResponse { result, .. } => {
                // F-901:provider usage 真值采集（对照 tiktoken 估算）
                let v = evorule_reactor::tcb_to_serde(result);
                if let Some(ut) = extract_usage_truth(&v) {
                    face.usage_trues.push(ut);
                }
            }
            evorule_reactor::Fact::Stable { .. } => face.stable_entries += 1,
            _ => {}
        }
    }
    face.wire_snapshots = snaps.len() as u64;
    Ok((face, snaps))
}

/// 从 inspector WAL 抽取 wire 面（轮次数）。
pub fn extract_wire_face(wal: &Path) -> Result<WireFace, crate::InspectorError> {
    let file = std::fs::File::open(wal)
        .map_err(|e| crate::InspectorError::Io(format!("inspector WAL 读取失败 {wal:?}: {e}")))?;
    let mut rounds = 0u64;
    for line in std::io::BufReader::new(file).lines() {
        let line = line.map_err(|e| crate::InspectorError::Io(format!("读行失败: {e}")))?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(w) = serde_json::from_str::<WalLine>(&line) else {
            continue;
        };
        if matches!(w.record, InspectorRecord::Snapshot(_)) {
            rounds += 1;
        }
    }
    Ok(WireFace {
        observed_rounds: rounds,
    })
}

// ---------- 八层启发式归因 ----------

/// 对一份 wire（按消息序）做八层归因（格式特征启发式，T-2 一期口径）。
pub fn attribute(
    messages: &[FaceMessage],
    tools: usize,
    tokens_est_fn: impl Fn(&str) -> u64,
) -> Vec<AttributedMessage> {
    let mut out = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let (layer, section) = attribute_one(m, i, messages.len());
        let tokens = tokens_est_fn(&m.content);
        out.push(AttributedMessage {
            layer,
            role: m.role.clone(),
            index: i,
            chars: m.content.chars().count(),
            tokens_est: tokens,
            section,
        });
    }
    if tools > 0 {
        // L5 能力层以独立行报告（tools 定义不入 messages，单列计数）
        out.push(AttributedMessage {
            layer: Layer::Capability,
            role: "tools".into(),
            index: usize::MAX,
            chars: 0,
            tokens_est: 0,
            section: Some(format!("{tools} tool definitions")),
        });
    }
    out
}

fn attribute_one(m: &FaceMessage, index: usize, total: usize) -> (Layer, Option<String>) {
    if m.role == "tool" {
        return (Layer::Observation, None);
    }
    if m.role == "system" {
        // 分区标记（00 基线 §三.2：## Stable Facts / ## Previous Sessions / ## Relevant Events）
        for (marker, sec) in [
            ("## Stable Facts", "Stable Facts"),
            ("## Previous Sessions", "Previous Sessions"),
            ("## Relevant Events", "Relevant Events"),
        ] {
            if m.content.contains(marker) {
                return (Layer::Knowledge, Some(sec.to_owned()));
            }
        }
        // hint/摘要替换槽（context_window trim hint 形态）
        if m.content.starts_with("[earlier ") && m.content.contains("trimmed") {
            return (Layer::History, Some("hint/summary".into()));
        }
        return (Layer::SystemCore, None);
    }
    if m.role == "user" {
        // 首条 user（system 之后第一条）= goal；后续 user = 历史轮
        return if index == 1 || (index == 0 && total == 1) {
            (Layer::Goal, None)
        } else {
            (Layer::History, None)
        };
    }
    (Layer::History, None)
}

// ---------- 检查器 ----------

/// 万分比（整数运算，分母 0 → 0；u128 防溢出——与 pressure 同款纪律）。
fn permyriad(part: u64, total: u64) -> u64 {
    (u128::from(part) * 10_000 / u128::from(total.max(1))) as u64
}

/// I5 摘要保真（R3 方案 B 落地后可验证）：对照约束源与每代摘要文本，
/// 检测压缩丢约束（参考档失败模式 F2）。缺约束 = Critical（对照 dropped
/// 原文核验——persist 全文在链，可定位丢失内容）。
pub fn i5_summary_fidelity(
    landings: &[RollingSummaryLanding],
    constraints: &ConstraintSource,
) -> Vec<CheckFinding> {
    let mut out = Vec::new();
    for landing in landings {
        out.push(CheckFinding {
            checker: "i5_summary_fidelity",
            severity: Severity::Info,
            message: format!(
                "滚动摘要 gen={} frozen={}->{} 已落链——I5 保真可验证（对照约束源）",
                landing.gen, landing.frozen_len_before, landing.frozen_len_after
            ),
        });
        for p in &constraints.phrases {
            if !landing.summary_text.contains(p.as_str()) {
                out.push(CheckFinding {
                    checker: "i5_summary_fidelity",
                    severity: Severity::Critical,
                    message: format!(
                        "摘要丢失约束（I5/F2）：gen={} 摘要缺「{}」——对照 dropped 原文核验（persist 全文在链）",
                        landing.gen, p
                    ),
                });
            }
        }
    }
    out
}

/// 检查器入参：归因结果 + 原始消息（供内容级检查）。
pub fn run_checks(
    messages: &[FaceMessage],
    attribution: &[AttributedMessage],
    constraints: &ConstraintSource,
    landings: &[RollingSummaryLanding],
) -> Vec<CheckFinding> {
    let mut out = Vec::new();
    let total_tokens: u64 = attribution.iter().map(|a| a.tokens_est).sum();

    // 1) token 归因分布（Info：报告面）
    let mut by_layer: BTreeMap<&str, u64> = BTreeMap::new();
    for a in attribution {
        if a.role != "tools" {
            *by_layer.entry(a.layer.as_str()).or_default() += a.tokens_est;
        }
    }
    for (layer, tokens) in &by_layer {
        let share = permyriad(*tokens, total_tokens);
        out.push(CheckFinding {
            checker: "token_attribution",
            severity: Severity::Info,
            message: format!("{layer}: {tokens} tokens ({share}‱)"),
        });
    }

    // 2) 工具膨胀（L7 占比 > 50% → Warning；单条 > 24_000 字符 → Warning）
    let l7 = attribution
        .iter()
        .filter(|a| a.layer == Layer::Observation)
        .map(|a| a.tokens_est)
        .sum::<u64>();
    let l7_share = permyriad(l7, total_tokens);
    if l7_share > 5_000 {
        out.push(CheckFinding {
            checker: "tool_bloat",
            severity: Severity::Warning,
            message: format!(
                "L7 观察 token 占比 {l7_share}‱ > 50%——工具输出淹没（建议结构化解析后再喂）"
            ),
        });
    }
    for m in messages {
        if m.role == "tool" && m.content.chars().count() > 24_000 {
            out.push(CheckFinding {
                checker: "tool_bloat",
                severity: Severity::Warning,
                message: format!(
                    "单条 tool_result {} 字符 > 24_000（回喂截断阈值 TOOL_RESULT_MAX_CHARS 一半）",
                    m.content.chars().count()
                ),
            });
        }
    }

    // 3) 重复检测（L7/L8 跨消息内容哈希精确重复）
    let mut seen: HashSet<u64> = HashSet::new();
    for m in messages {
        if m.role == "system" {
            continue; // system 同文逐轮重复属正常形态（豁免），重复检测面向 L7/L8 与后续 user
        }
        let h = evorule_hash::digest(m.content.as_bytes());
        let key = u64::from_be_bytes(h.as_bytes()[..8].try_into().unwrap_or([0; 8]));
        if !seen.insert(key) {
            out.push(CheckFinding {
                checker: "repetition",
                severity: Severity::Warning,
                message: format!(
                    "{} 角色消息内容与前文完全重复（前 8 字符：{}）",
                    m.role,
                    &m.content[..m.content.chars().count().min(8)]
                ),
            });
        }
    }

    // 4) 约束存在性 + 生存率（shadow 策略 #1：只记录不执行）
    // 5) I1 字面级：约束短语在 L7/L8 中出现否定式对冲（"X 禁止" vs "请 X"）
    out.extend(constraint_checks(messages, constraints));

    // 6) I2 字面级：同层重复矛盾（同层内同一关键词的正反极性共存）——保守起步：
    //    与 repetition 检查互补，此处仅报告同层完全同文重复（已在 repetition 覆盖），占位说明。

    // 7) I5 摘要保真（R3 落链后可验证；无落链 = 报告能力边界）
    if landings.is_empty() {
        out.push(CheckFinding {
            checker: "i5_summary_fidelity",
            severity: Severity::Info,
            message: "链上无滚动摘要落链记录——I5 摘要保真暂不可验证（R3 方案 B 未启用或无摘要代）"
                .to_owned(),
        });
    } else {
        out.extend(i5_summary_fidelity(landings, constraints));
    }
    out
}

/// 约束面检查（存在性/生存率 + I1 字面级对冲）。
fn constraint_checks(
    messages: &[FaceMessage],
    constraints: &ConstraintSource,
) -> Vec<CheckFinding> {
    let mut out = Vec::new();
    let latest_wire = messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let (present, missing) = count_survival(&constraints.phrases, &latest_wire);
    if !constraints.phrases.is_empty() {
        let total = constraints.phrases.len();
        out.push(CheckFinding {
            checker: "constraint_presence",
            severity: if missing.is_empty() {
                Severity::Info
            } else {
                Severity::Warning
            },
            message: format!(
                "约束生存率 {present}/{total}（来源={}）；缺失：{}",
                constraints.source,
                missing.join(" | ")
            ),
        });
    }

    for phrase in &constraints.phrases {
        let core = strip_modals(phrase);
        if core.is_empty() {
            continue;
        }
        for m in messages {
            if m.role == "system" {
                continue;
            }
            if let Some(hit) = literal_negation(&core, &m.content) {
                out.push(CheckFinding {
                    checker: "i1_authority_literal",
                    severity: Severity::Warning,
                    message: format!(
                        "约束「{core}」在 {} 角色消息中疑似对冲（{}）——字面级启发，语义级留离线",
                        m.role, hit
                    ),
                });
            }
        }
    }
    out
}

fn count_survival(phrases: &[String], wire: &str) -> (usize, Vec<String>) {
    let mut present = 0;
    let mut missing = Vec::new();
    for p in phrases {
        if wire.contains(p.as_str()) {
            present += 1;
        } else {
            missing.push(p.clone());
        }
    }
    (present, missing)
}

fn strip_modals(phrase: &str) -> String {
    let mut s = phrase.to_owned();
    for m in [
        "禁止",
        "不得",
        "不能",
        "必须",
        "永远",
        "一定要",
        "must",
        "never",
        "always",
        "请",
    ] {
        s = s.replace(m, "");
    }
    s.trim().trim_start_matches(['，', '。', '、']).to_owned()
}

/// 字面级否定式对冲检测（保守：核心词出现且同一句含"可以/允许/请随意"）。
fn literal_negation(core: &str, content: &str) -> Option<String> {
    if !content.contains(core) {
        return None;
    }
    for sentence in content.split(['。', '\n', '；']) {
        let lower = sentence.to_lowercase();
        if lower.contains(core)
            && (lower.contains("可以") || lower.contains("允许") || lower.contains("随意"))
        {
            return Some(sentence.trim().to_owned());
        }
    }
    None
}
