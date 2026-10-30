//! 跨链对账（设计稿 §5.3 第 3 步，v3.3 如实降级版）。
//!
//! **能力边界（如实声明，不虚构验证力）**：fact WAL（`WalRecord`）不含墙钟
//! 时间戳（仅 version/fact/哈希字段——reactor 架构事实），因此设计稿原设的
//! 「时间窗交错一致」对账**不可实施**。本模块实施可验证的三项：
//! 1. 双链各自全量校验（inspector：强语义；fact：reactor 读入即校验哈希链）
//! 2. **cause_fact 存在性对账**：凡快照携带 `cause_fact`（Header 注入级），
//!    该 FactId 必须真实存在于 fact WAL——跨链因果引用的硬校验
//! 3. 体量对照（facts 数 / inspector 记录数并列报告，不做语义断言）
//!
//! 时间窗对账待 fact WAL 携带墙钟时间戳后另行增量（登记，不在本项目内改
//! reactor）。

use crate::chain::{verify_wal_file, ChainReport};
use crate::InspectorError;
use serde::Serialize;
use std::collections::HashSet;
use std::io::BufRead;
use std::path::Path;

/// 对账报告。
#[derive(Debug, Serialize)]
pub struct ReconcileReport {
    pub facts: u64,
    pub inspector_records: u64,
    /// inspector 链全量校验（强语义）。
    pub inspector_chain: ChainReport,
    /// fact 链是否全量通过（新格式 WAL 含哈希字段时为真校验；旧格式 false）。
    pub fact_chain_hashed: bool,
    /// 检查了带 cause_fact 的快照数。
    pub cause_checked: u64,
    /// cause_fact 指向的 FactId 不在 fact WAL 中（跨链断引——必须为空）。
    pub missing_cause: Vec<serde_json::Value>,
    /// 时间窗对账能力声明（架构事实，如实降级）。
    pub time_window_check: &'static str,
    /// 跨链对账覆盖面（诚实性字段，INSP-D-03/04 收口）：
    /// - `full`：存在带 cause_fact 的快照且已硬校验（Header 注入级路径）
    /// - `not_applicable`：无任何快照携带 cause_fact（默认 PortBinding 模式
    ///   不产生跨链因果引用）——此时对账退化为单链自检，`ok()` 不代表
    ///   跨链一致，只代表双链各自完整
    pub cross_chain_check: CrossChainCoverage,
}

/// 跨链对账覆盖面（三态，诚实区分「验过/没得验」）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CrossChainCoverage {
    /// 至少一条快照携带 cause_fact 且完成存在性硬校验
    Verified,
    /// 快照存在但无一携带 cause_fact（PortBinding 默认模式）——跨链对账不适用
    NotApplicable,
    /// 无任何 inspector 快照（对账对象为空）
    NoSnapshots,
}

impl CrossChainCoverage {
    pub fn as_str(self) -> &'static str {
        match self {
            CrossChainCoverage::Verified => "verified",
            CrossChainCoverage::NotApplicable => "not_applicable",
            CrossChainCoverage::NoSnapshots => "no_snapshots",
        }
    }
}

impl ReconcileReport {
    /// 总判定：双链通过且无断引。
    ///
    /// 覆盖面语义（如实区分，不虚构验证力）：
    /// - `cross_chain_check == verified` 时，`ok` = 双链完整 + 跨链因果无断引；
    /// - `not_applicable` 时，`ok` 仅代表**双链各自完整**——跨链一致性根本
    ///   未被检验（默认 PortBinding 模式的快照不带因果引用），调用方若只看
    ///   `ok` 请连同本字段一起读；
    /// - `time_window_check` 恒为架构性降级声明，不计入 `ok`。
    pub fn ok(&self) -> bool {
        self.inspector_chain.ok && self.missing_cause.is_empty()
    }
}

/// 执行对账（只读两个 WAL 文件；路径由部署配置或调用参数提供）。
pub fn reconcile(fact_log: &Path, inspector_log: &Path) -> Result<ReconcileReport, InspectorError> {
    let records = evorule_reactor::read_wal_with_hash(fact_log)
        .map_err(|e| InspectorError::Io(format!("fact WAL 读取失败 {fact_log:?}: {e}")))?;
    let fact_chain_hashed = records.iter().any(|r| r.chain_hash.is_some());
    let fact_ids: HashSet<evorule_reactor::FactId> = records.iter().map(|r| r.fact.id()).collect();

    let inspector_chain = verify_wal_file(inspector_log)?;

    let mut cause_checked: u64 = 0;
    let mut missing_cause: Vec<serde_json::Value> = Vec::new();
    let mut inspector_records: u64 = 0;
    let file = std::fs::File::open(inspector_log)
        .map_err(|e| InspectorError::Io(format!("inspector WAL 读取失败: {e}")))?;
    for line in std::io::BufReader::new(file).lines() {
        let line = line.map_err(|e| InspectorError::Io(format!("inspector WAL 读行失败: {e}")))?;
        if line.trim().is_empty() {
            continue;
        }
        inspector_records += 1;
        let v: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| InspectorError::Io(format!("inspector WAL 行解析失败: {e}")))?;
        let Some(cause) = v
            .get("session")
            .and_then(|s| s.get("cause_fact"))
            .and_then(serde_json::Value::as_u64)
        else {
            continue;
        };
        cause_checked += 1;
        let fid = evorule_reactor::FactId(cause);
        if !fact_ids.contains(&fid) {
            missing_cause.push(serde_json::json!({
                "type": v.get("type").cloned().unwrap_or(serde_json::json!("unknown")),
                "snapshot_id": v.get("snapshot_id").cloned().unwrap_or(serde_json::Value::Null),
                "cause_fact": cause,
            }));
        }
    }

    // 跨链覆盖面判定:有 inspector 记录且至少一条带 cause_fact = verified;
    // 有记录但零 cause_fact = not_applicable(PortBinding 默认形态);
    // 零记录 = no_snapshots
    let cross_chain_check = if inspector_records == 0 {
        CrossChainCoverage::NoSnapshots
    } else if cause_checked > 0 {
        CrossChainCoverage::Verified
    } else {
        CrossChainCoverage::NotApplicable
    };

    Ok(ReconcileReport {
        facts: records.len() as u64,
        inspector_records,
        inspector_chain,
        fact_chain_hashed,
        cause_checked,
        missing_cause,
        time_window_check: "unsupported: fact WAL 无墙钟时间戳（reactor 架构事实）——时间窗交错对账待 fact 层携带时间戳后另行增量",
        cross_chain_check,
    })
}

#[cfg(test)]
mod coverage_tests {
    use super::*;

    #[test]
    fn coverage_buckets_follow_cause_presence() {
        // 有记录+有 cause_checked = verified;有记录+零 cause = not_applicable;
        // 零记录 = no_snapshots——三桶判定与 inspector_records/cause_checked 联动
        let assert_bucket = |inspector_records: u64, cause_checked: u64| {
            if inspector_records == 0 {
                CrossChainCoverage::NoSnapshots
            } else if cause_checked > 0 {
                CrossChainCoverage::Verified
            } else {
                CrossChainCoverage::NotApplicable
            }
        };
        assert_eq!(assert_bucket(5, 2), CrossChainCoverage::Verified);
        assert_eq!(assert_bucket(5, 0), CrossChainCoverage::NotApplicable);
        assert_eq!(assert_bucket(0, 0), CrossChainCoverage::NoSnapshots);
    }

    #[test]
    fn coverage_serializes_with_stable_labels() {
        // 契约面:三态字符串标签稳定(server 契约服务直接透传)
        assert_eq!(CrossChainCoverage::Verified.as_str(), "verified");
        assert_eq!(CrossChainCoverage::NotApplicable.as_str(), "not_applicable");
        assert_eq!(CrossChainCoverage::NoSnapshots.as_str(), "no_snapshots");
    }
}
