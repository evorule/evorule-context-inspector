//! InspectorLog —— 独立 BLAKE3 审计链（设计稿 §5.2；对齐 reactor `FactsLog`
//! 的 `append(&self)` 内部锁 + **WAL write-ahead** 模式）。
//!
//! - **链步单一真相源**：`evorule_reactor::chain_step(prev, content)`
//!   （`blake3(prev + content)`），genesis=`"genesis"`（hash.rs:449 同款）；
//!   禁自写第二套链步实现（§5.1 增强 2）。
//! - **记录哈希**：Snapshot → `skeleton_hash`（存储结构可重算 → 链验证强语义，
//!   §2.5；链上只覆盖骨架=设计口径）；Observation →
//!   BLAKE3(canonical_json(observation))（全量覆盖）。
//! - **WAL 先写**：内存更新**前**写 JSONL 行（seq/prev_hash/record_hash/
//!   chain_hash/record）并 flush——对齐 `append_record_with_hash` 纪律；
//!   WAL 失败 = 拒收且内存不变（R4 fail-fast）。
//! - **热窗口**：内存只保最近 K 条（默认 1024）供查询，全量事实以 WAL 为准；
//!   `recover` 从 WAL 重放并逐条校验，损坏即报错定位（验收 #12）。
//! - 单链全局有序：所有 session 的记录进同一条链；`session_index`
//!   为热窗口内的二级索引（设计稿 §5.2）。

use crate::model::{ContextSnapshot, ResponseObservation};
use crate::{InspectorError, SessionId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use uuid::Uuid;

/// 链初值（对齐 reactor hash.rs genesis）。
pub const GENESIS: &str = "genesis";

/// 热窗口默认容量（§5.2：内存只保最近 K 条，默认 1024，可配）。
pub const HOT_WINDOW_DEFAULT: usize = 1024;

/// 一条审计记录：请求侧快照或响应侧观测。
/// （变体尺寸差 allow：记录全程值语义流经单写锁 append，装箱徒增每次
/// 追加一次堆分配，336B 栈占用可接受。）
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InspectorRecord {
    Snapshot(ContextSnapshot),
    Observation(ResponseObservation),
}

impl InspectorRecord {
    /// 记录哈希（§5.2 + v3.2 偏差③，项目方 2026-10-02 裁定「改」）：
    /// Snapshot → **BLAKE3(canonical_json(全快照))**——链覆盖扩展到 composition/
    /// model/session 等全部字段（原稿仅骨架，composition 篡改不破链的完整性
    /// 缺口经审批收口）；`skeleton_hash` 字段保留，verify 线内另作强语义锚点
    /// 校验（存储骨架重算 == 字段）。诚实记录的 canonical 哈希与追加时恒等。
    /// Observation → BLAKE3(canonical_json(observation)) 全量覆盖。
    pub fn record_hash(&self) -> Result<crate::HashHex, InspectorError> {
        match self {
            InspectorRecord::Snapshot(s) => {
                crate::canonical::canonical_json(s).map(|s| evorule_hash::digest(s.as_bytes()))
            }
            InspectorRecord::Observation(o) => {
                crate::canonical::canonical_json(o).map(|s| evorule_hash::digest(s.as_bytes()))
            }
        }
    }

    /// 记录自身 ID（append 的返回值；不误用 FactId——v3 §5.2）。
    pub fn own_id(&self) -> Uuid {
        match self {
            InspectorRecord::Snapshot(s) => s.snapshot_id,
            InspectorRecord::Observation(o) => o.observation_id,
        }
    }

    /// 快照记录的会话 ID（观测记录 None——其归属经 `response_to` 配对解析）。
    pub fn snapshot_session(&self) -> Option<SessionId> {
        match self {
            InspectorRecord::Snapshot(s) => s.session.session_id,
            InspectorRecord::Observation(_) => None,
        }
    }
}

/// 链校验报告（§5.3）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChainReport {
    pub records: u64,
    pub ok: bool,
    /// 首个断裂记录的序号（1 起；ok=true 时为 None）。
    pub first_bad_seq: Option<u64>,
    pub reason: Option<String>,
}

/// 热窗口条目：带三段哈希，使内存窗口可独立复验（非纯重放）。
#[derive(Debug)]
struct HotEntry {
    seq: u64,
    record: InspectorRecord,
    prev_hash: crate::HashHex,
    record_hash: crate::HashHex,
    chain_hash: crate::HashHex,
}

#[derive(Debug)]
struct Inner {
    last_hash: crate::HashHex,
    seq: u64,
    hot: VecDeque<HotEntry>,
    /// 热窗口内 session_id → 快照 Uuid 列表（观测经配对解析，不入此索引）。
    session_index: HashMap<SessionId, Vec<Uuid>>,
    wal: Option<std::fs::File>,
    wal_path: Option<PathBuf>,
}

/// 独立审计链日志（`append(&self)` 内部写锁，多会话共享安全）。
#[derive(Debug)]
pub struct InspectorLog {
    inner: RwLock<Inner>,
}

/// WAL JSONL 行（读取态，拥有 record；crate 内复用——reconcile 扫描）。
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct WalLine {
    pub(crate) seq: u64,
    pub(crate) prev_hash: crate::HashHex,
    pub(crate) record_hash: crate::HashHex,
    pub(crate) chain_hash: crate::HashHex,
    #[serde(flatten)]
    pub(crate) record: InspectorRecord,
}

/// WAL JSONL 行（写入态，借位序列化——append 内 record 所有权保留给内存）。
#[derive(Debug, Serialize)]
struct WalLineRef<'a> {
    seq: u64,
    prev_hash: &'a str,
    record_hash: &'a str,
    chain_hash: &'a str,
    #[serde(flatten)]
    record: &'a InspectorRecord,
}

/// 单行校验（recover 与 verify_wal_file 共用的判定核）。
fn verify_line(
    seq: u64,
    expected_prev: &str,
    wal_line: &WalLine,
) -> Result<crate::HashHex, InspectorError> {
    if wal_line.seq != seq {
        return Err(InspectorError::WalCorrupt {
            seq,
            reason: format!("seq 不连续：期望 {seq}，实得 {}", wal_line.seq),
        });
    }
    if wal_line.prev_hash != expected_prev {
        return Err(InspectorError::WalCorrupt {
            seq,
            reason: format!(
                "prev_hash 不接续：期望 {expected_prev}，实得 {}",
                wal_line.prev_hash
            ),
        });
    }
    // 强语义锚点：Snapshot 的 skeleton_hash 字段必须等于存储骨架重算值
    if let InspectorRecord::Snapshot(s) = &wal_line.record {
        let anchor = s.skeleton.skeleton_hash()?;
        if anchor != s.skeleton_hash {
            return Err(InspectorError::WalCorrupt {
                seq,
                reason: format!(
                    "skeleton_hash 锚点不符：重算 {anchor} ≠ 字段 {}",
                    s.skeleton_hash
                ),
            });
        }
    }
    // 全覆盖记录哈希：从存储记录重算比对（composition/model 等一并入覆盖）
    let recomputed = wal_line.record.record_hash()?;
    if recomputed != wal_line.record_hash {
        return Err(InspectorError::WalCorrupt {
            seq,
            reason: format!(
                "记录哈希不符：重算 {recomputed} ≠ 落账 {}",
                wal_line.record_hash
            ),
        });
    }
    let expect = evorule_reactor::chain_step(&wal_line.prev_hash, &wal_line.record_hash);
    if expect != wal_line.chain_hash {
        return Err(InspectorError::WalCorrupt {
            seq,
            reason: format!("链步不符：重算 {expect} ≠ 落账 {}", wal_line.chain_hash),
        });
    }
    Ok(wal_line.chain_hash.clone())
}

impl InspectorLog {
    /// 纯内存模式（合法形态：`--inspector-log-dir` 未传 = 不持久化，§1 决策 7）。
    pub fn in_memory() -> Self {
        Self {
            inner: RwLock::new(Inner {
                last_hash: GENESIS.to_owned(),
                seq: 0,
                hot: VecDeque::new(),
                session_index: HashMap::new(),
                wal: None,
                wal_path: None,
            }),
        }
    }

    /// 打开（或恢复）落盘模式：文件存在则 recover，否则新建。
    pub fn open(wal_path: impl AsRef<Path>) -> Result<Self, InspectorError> {
        let path = wal_path.as_ref();
        if path.exists() {
            Self::recover(path)
        } else {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| InspectorError::Io(format!("创建目录失败 {parent:?}: {e}")))?;
            }
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| InspectorError::Io(format!("打开 WAL 失败 {path:?}: {e}")))?;
            Ok(Self {
                inner: RwLock::new(Inner {
                    last_hash: GENESIS.to_owned(),
                    seq: 0,
                    hot: VecDeque::new(),
                    session_index: HashMap::new(),
                    wal: Some(file),
                    wal_path: Some(path.to_path_buf()),
                }),
            })
        }
    }

    /// 从 WAL 重放恢复：逐条校验（强语义 + 链接），损坏即拒载定位（R4）。
    pub fn recover(path: impl AsRef<Path>) -> Result<Self, InspectorError> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| InspectorError::Io(format!("打开 WAL 失败 {path:?}: {e}")))?;
        let mut last_hash = GENESIS.to_owned();
        let mut seq: u64 = 0;
        let mut hot: VecDeque<HotEntry> = VecDeque::new();
        let mut session_index: HashMap<SessionId, Vec<Uuid>> = HashMap::new();
        for line in std::io::BufReader::new(file).lines() {
            let line = line.map_err(|e| InspectorError::WalCorrupt {
                seq: seq + 1,
                reason: format!("读行失败: {e}"),
            })?;
            if line.trim().is_empty() {
                continue;
            }
            seq += 1;
            let wal_line: WalLine =
                serde_json::from_str(&line).map_err(|e| InspectorError::WalCorrupt {
                    seq,
                    reason: format!("JSON 解析失败: {e}"),
                })?;
            let chain_hash = verify_line(seq, &last_hash, &wal_line)?;
            if let InspectorRecord::Snapshot(s) = &wal_line.record {
                if let Some(sid) = s.session.session_id {
                    session_index.entry(sid).or_default().push(s.snapshot_id);
                }
            }
            hot.push_back(HotEntry {
                seq,
                prev_hash: wal_line.prev_hash,
                record_hash: wal_line.record_hash,
                chain_hash: chain_hash.clone(),
                record: wal_line.record,
            });
            if hot.len() > HOT_WINDOW_DEFAULT {
                if let Some(evicted) = hot.pop_front() {
                    evict_session(&mut session_index, &evicted.record);
                }
            }
            last_hash = chain_hash;
        }
        let file = std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| InspectorError::Io(format!("重开 WAL 失败 {path:?}: {e}")))?;
        Ok(Self {
            inner: RwLock::new(Inner {
                last_hash,
                seq,
                hot,
                session_index,
                wal: Some(file),
                wal_path: Some(path.to_path_buf()),
            }),
        })
    }

    /// 追加一条记录（§5.2）：WAL 先写并 flush（失败=拒收，内存不变），
    /// 成功才更新内存；返回记录自身 ID。
    pub fn append(&self, record: InspectorRecord) -> Result<Uuid, InspectorError> {
        let mut inner = self.inner.write().map_err(|_| poisoned())?;
        let own_id = record.own_id();
        let session = record.snapshot_session();
        let prev_hash = inner.last_hash.clone();
        let new_seq = inner.seq + 1;
        let record_hash = record.record_hash()?;
        let chain_hash = evorule_reactor::chain_step(&prev_hash, &record_hash);
        // WAL write-ahead：先落盘（含三段哈希字段）+ flush；块内只用 wal 与局部量
        if let Some(wal) = inner.wal.as_mut() {
            let line = WalLineRef {
                seq: new_seq,
                prev_hash: &prev_hash,
                record_hash: &record_hash,
                chain_hash: &chain_hash,
                record: &record,
            };
            let json = serde_json::to_string(&line)
                .map_err(|e| InspectorError::Io(format!("WAL 行序列化失败: {e}")))?;
            writeln!(wal, "{json}")
                .map_err(|e| InspectorError::Io(format!("WAL 写入失败: {e}")))?;
            wal.flush()
                .map_err(|e| InspectorError::Io(format!("WAL flush 失败: {e}")))?;
        }
        // WAL 成功（或纯内存模式）→ 更新内存
        if let Some(sid) = session {
            inner.session_index.entry(sid).or_default().push(own_id);
        }
        inner.seq = new_seq;
        inner.last_hash = chain_hash.clone();
        inner.hot.push_back(HotEntry {
            seq: new_seq,
            record,
            prev_hash,
            record_hash,
            chain_hash,
        });
        if inner.hot.len() > HOT_WINDOW_DEFAULT {
            if let Some(evicted) = inner.hot.pop_front() {
                evict_session(&mut inner.session_index, &evicted.record);
            }
        }
        Ok(own_id)
    }

    /// 链头（G-A1 锚点发布与对外可见性的入口，§5.1）。
    pub fn anchor_head(&self) -> Result<crate::HashHex, InspectorError> {
        Ok(self.inner.read().map_err(|_| poisoned())?.last_hash.clone())
    }

    /// 已落账记录总数（含热窗口之外）。
    pub fn total_records(&self) -> Result<u64, InspectorError> {
        Ok(self.inner.read().map_err(|_| poisoned())?.seq)
    }

    /// 热窗口记录只读快照（查询面用；全量事实以 WAL 为准，§5.2）。
    pub fn hot_records(&self) -> Result<Vec<(u64, InspectorRecord)>, InspectorError> {
        let inner = self.inner.read().map_err(|_| poisoned())?;
        Ok(inner
            .hot
            .iter()
            .map(|e| (e.seq, e.record.clone()))
            .collect())
    }

    /// 按 ID 查热窗口内快照。
    pub fn snapshot_by_id(&self, id: Uuid) -> Result<Option<ContextSnapshot>, InspectorError> {
        let inner = self.inner.read().map_err(|_| poisoned())?;
        Ok(inner.hot.iter().find_map(|e| match &e.record {
            InspectorRecord::Snapshot(s) if s.snapshot_id == id => Some(s.clone()),
            _ => None,
        }))
    }

    /// 热窗口内某会话的快照 ID 列表（二级索引查询，§5.2）。
    pub fn snapshot_ids_of_session(&self, session: SessionId) -> Result<Vec<Uuid>, InspectorError> {
        let inner = self.inner.read().map_err(|_| poisoned())?;
        Ok(inner
            .session_index
            .get(&session)
            .cloned()
            .unwrap_or_default())
    }

    /// 全量校验：落盘模式重读 WAL（权威，含热窗口之外历史，§5.3 第 2 步）；
    /// 纯内存模式复验热窗口（逐条重算记录哈希 + 三段哈希链接）。
    pub fn verify_chain(&self) -> Result<ChainReport, InspectorError> {
        let wal_path = self.inner.read().map_err(|_| poisoned())?.wal_path.clone();
        if let Some(path) = wal_path {
            return verify_wal_file(&path);
        }
        let inner = self.inner.read().map_err(|_| poisoned())?;
        let mut expected_prev = GENESIS.to_owned();
        for entry in &inner.hot {
            let recomputed = entry.record.record_hash()?;
            if recomputed != entry.record_hash
                || entry.prev_hash != expected_prev
                || evorule_reactor::chain_step(&entry.prev_hash, &entry.record_hash)
                    != entry.chain_hash
            {
                return Ok(ChainReport {
                    records: entry.seq.saturating_sub(1),
                    ok: false,
                    first_bad_seq: Some(entry.seq),
                    reason: Some("热窗口复验失败（内存链断裂）".into()),
                });
            }
            expected_prev = entry.chain_hash.clone();
        }
        let head_ok = inner.hot.back().map(|e| e.chain_hash.clone())
            == Some(inner.last_hash.clone())
            || inner.seq == 0;
        Ok(ChainReport {
            records: inner.seq,
            ok: head_ok,
            first_bad_seq: None,
            reason: if head_ok {
                None
            } else {
                Some("链头与热窗口末条不一致".into())
            },
        })
    }
}

fn evict_session(index: &mut HashMap<SessionId, Vec<Uuid>>, record: &InspectorRecord) {
    if let InspectorRecord::Snapshot(s) = record {
        if let Some(sid) = s.session.session_id {
            if let Some(v) = index.get_mut(&sid) {
                v.retain(|id| *id != s.snapshot_id);
                if v.is_empty() {
                    index.remove(&sid);
                }
            }
        }
    }
}

fn poisoned() -> InspectorError {
    InspectorError::Io("InspectorLog 锁中毒（并发写线程 panic）".into())
}

/// 独立 WAL 文件全量校验（verify_chain 的落盘路径；CLI `verify-chain
/// --inspector-log` 复用同一入口，§5.3）。
pub fn verify_wal_file(path: impl AsRef<Path>) -> Result<ChainReport, InspectorError> {
    let file = std::fs::File::open(path.as_ref())
        .map_err(|e| InspectorError::Io(format!("打开 WAL 失败: {e}")))?;
    let mut last = GENESIS.to_owned();
    let mut seq: u64 = 0;
    for line in std::io::BufReader::new(file).lines() {
        let line = line.map_err(|e| InspectorError::WalCorrupt {
            seq: seq + 1,
            reason: format!("读行失败: {e}"),
        })?;
        if line.trim().is_empty() {
            continue;
        }
        seq += 1;
        let wal_line: WalLine =
            serde_json::from_str(&line).map_err(|e| InspectorError::WalCorrupt {
                seq,
                reason: format!("JSON 解析失败: {e}"),
            })?;
        last = match verify_line(seq, &last, &wal_line) {
            Ok(ch) => ch,
            // 校验失败 → 失败报告（定位序号）；Err 仅保留给 I/O 故障
            Err(e) => {
                return Ok(ChainReport {
                    records: seq,
                    ok: false,
                    first_bad_seq: Some(seq),
                    reason: Some(e.to_string()),
                });
            }
        };
    }
    Ok(ChainReport {
        records: seq,
        ok: true,
        first_bad_seq: None,
        reason: None,
    })
}

/// 校验失败希望拿到定位信息时的形态（CLI 报错用）。
pub fn verify_wal_file_detailed(path: impl AsRef<Path>) -> ChainReport {
    match verify_wal_file(path) {
        Ok(report) => report,
        Err(e) => ChainReport {
            records: 0,
            ok: false,
            first_bad_seq: match &e {
                InspectorError::WalCorrupt { seq, .. } => Some(*seq),
                _ => None,
            },
            reason: Some(e.to_string()),
        },
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use crate::model::{
        Composition, ContextSkeleton, ContextSnapshot, MessageEntry, ResponseObservation,
        SessionLink, ToolEntry, UsageInfo,
    };
    use crate::TokenSource;
    use crate::{BindingKind, ContentKind, ProviderKind};

    fn snapshot(seq_id: u64, session: Option<u64>, msg: &[u8]) -> ContextSnapshot {
        let skeleton = ContextSkeleton::new(
            ProviderKind::DeepSeek,
            "deepseek-chat",
            b"sys",
            vec![ToolEntry::new("t", b"schema")],
            vec![MessageEntry::new("user", ContentKind::Text, msg, 0)],
        );
        let composition = Composition::new(3, 5, 20, 0, 1, 1, TokenSource::Estimated).unwrap();
        ContextSnapshot::new(
            Uuid::from_u128(u128::from(seq_id)),
            SessionLink::new(session, BindingKind::PortBinding, None),
            1_000 + seq_id,
            ProviderKind::DeepSeek,
            "deepseek-chat",
            Some(128_000),
            composition,
            skeleton,
        )
        .unwrap()
    }

    fn observation(of: &ContextSnapshot, out: u64) -> ResponseObservation {
        ResponseObservation::new(
            Uuid::from_u128(of.snapshot_id.as_u128() + 10_000),
            of.snapshot_id,
            2_000,
            Some("stop".into()),
            UsageInfo {
                input_tokens: Some(28),
                output_tokens: Some(out),
                cached_tokens: None,
            },
            false,
            false,
        )
    }

    fn temp_wal(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir();
        dir.join(format!(
            "inspector_test_{}_{}.jsonl",
            tag,
            uuid::Uuid::new_v4().simple()
        ))
    }

    #[test]
    fn append_and_verify_wal_roundtrip() {
        let path = temp_wal("roundtrip");
        let log = InspectorLog::open(&path).unwrap();
        let s1 = snapshot(1, Some(7), b"hello");
        let id1 = log.append(InspectorRecord::Snapshot(s1.clone())).unwrap();
        assert_eq!(id1, s1.snapshot_id);
        let o1 = observation(&s1, 9);
        let id2 = log.append(InspectorRecord::Observation(o1)).unwrap();
        assert_eq!(log.total_records().unwrap(), 2);
        assert_eq!(log.anchor_head().unwrap().len(), 64);
        assert_ne!(log.anchor_head().unwrap(), GENESIS);
        let report = log.verify_chain().unwrap();
        assert!(report.ok, "reason={:?}", report.reason);
        assert_eq!(report.records, 2);
        // 首行 genesis 链接
        let first = std::fs::read_to_string(&path).unwrap();
        let first_line = first.lines().next().unwrap();
        assert!(first_line.contains(&format!("\"prev_hash\":\"{GENESIS}\"")));
        drop(log);
        let _ = std::fs::remove_file(&path);
        let _ = id2;
    }

    #[test]
    fn recover_replays_and_reindexes() {
        let path = temp_wal("recover");
        {
            let log = InspectorLog::open(&path).unwrap();
            log.append(InspectorRecord::Snapshot(snapshot(1, Some(7), b"m1")))
                .unwrap();
            log.append(InspectorRecord::Snapshot(snapshot(2, Some(9), b"m2")))
                .unwrap();
            let head = log.anchor_head().unwrap();
            assert_ne!(head, GENESIS);
        } // drop → 文件保留
        let log = InspectorLog::recover(&path).unwrap();
        assert_eq!(log.total_records().unwrap(), 2);
        let ids7 = log.snapshot_ids_of_session(7).unwrap();
        assert_eq!(ids7, vec![Uuid::from_u128(1)]);
        assert_eq!(
            log.snapshot_ids_of_session(9).unwrap(),
            vec![Uuid::from_u128(2)]
        );
        assert!(log.verify_chain().unwrap().ok);
        // 追加在恢复后继续链接
        log.append(InspectorRecord::Snapshot(snapshot(3, Some(7), b"m3")))
            .unwrap();
        assert_eq!(log.snapshot_ids_of_session(7).unwrap().len(), 2);
        assert!(log.verify_chain().unwrap().ok);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn recover_rejects_tampered_skeleton_strong_semantics() {
        let path = temp_wal("tamper_skeleton");
        {
            let log = InspectorLog::open(&path).unwrap();
            log.append(InspectorRecord::Snapshot(snapshot(1, None, b"original")))
                .unwrap();
        }
        // 篡改形态 = 伪造存储中的骨架哈希字段（原文不在 WAL——隐私设计只存哈希）：
        // content_hash 被换 → 骨架结构变 → skeleton_hash 重算 ≠ 落账 → 强语义识破
        let raw = std::fs::read_to_string(&path).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(raw.trim()).unwrap();
        // serde(flatten) 使 record 字段内联于顶层
        v["skeleton"]["messages"][0]["content_hash"] = serde_json::json!("00".repeat(32));
        std::fs::write(&path, serde_json::to_string(&v).unwrap() + "\n").unwrap();
        let err = InspectorLog::recover(&path).unwrap_err();
        match err {
            InspectorError::WalCorrupt { seq: 1, reason } => {
                // 锚点校验先命中（骨架伪造的第一道防线）；全量哈希随后兜底
                assert!(
                    reason.contains("skeleton_hash 锚点不符") || reason.contains("记录哈希不符"),
                    "reason={reason}"
                );
            }
            other => panic!("期望 WalCorrupt(seq=1)，实得 {other:?}"),
        }
        // 独立校验入口同样定位
        let report = verify_wal_file_detailed(&path);
        assert!(!report.ok);
        assert_eq!(report.first_bad_seq, Some(1));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn recover_rejects_broken_linkage_and_truncation() {
        // 断链：改 prev_hash 字段
        let path = temp_wal("linkage");
        {
            let log = InspectorLog::open(&path).unwrap();
            log.append(InspectorRecord::Snapshot(snapshot(1, None, b"a")))
                .unwrap();
            log.append(InspectorRecord::Snapshot(snapshot(2, None, b"b")))
                .unwrap();
        }
        let raw = std::fs::read_to_string(&path).unwrap();
        let mut lines: Vec<String> = raw.lines().map(str::to_owned).collect();
        // 第二行 prev_hash 指向 genesis（应为第一行 chain_hash）→ 链接断裂
        let second: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
        let mut broken = second.clone();
        broken["prev_hash"] = serde_json::json!(GENESIS);
        lines[1] = serde_json::to_string(&broken).unwrap();
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        let report = verify_wal_file_detailed(&path);
        assert!(!report.ok);
        assert_eq!(report.first_bad_seq, Some(2));
        assert!(report
            .reason
            .clone()
            .unwrap_or_default()
            .contains("prev_hash"));
        let _ = std::fs::remove_file(&path);

        // 截断：末行切半 → JSON 解析失败定位
        let path2 = temp_wal("truncated");
        {
            let log = InspectorLog::open(&path2).unwrap();
            log.append(InspectorRecord::Snapshot(snapshot(1, None, b"a")))
                .unwrap();
            log.append(InspectorRecord::Snapshot(snapshot(2, None, b"b")))
                .unwrap();
        }
        let raw = std::fs::read_to_string(&path2).unwrap();
        let cut = &raw[..raw.len() - 12];
        std::fs::write(&path2, cut).unwrap();
        let report = verify_wal_file_detailed(&path2);
        assert!(!report.ok);
        assert_eq!(report.first_bad_seq, Some(2));
        assert!(report.reason.unwrap_or_default().contains("JSON"));
        let _ = std::fs::remove_file(&path2);
    }

    #[test]
    fn composition_tampering_breaks_chain_full_coverage() {
        // v3.2 偏差③（项目方裁定「改」）：record_hash 覆盖全快照——
        // composition 篡改即使骨架不动也必须破链（原稿缺口已收口）。
        let path = temp_wal("composition_scope");
        {
            let log = InspectorLog::open(&path).unwrap();
            log.append(InspectorRecord::Snapshot(snapshot(1, None, b"m")))
                .unwrap();
        }
        // 直接改 JSONL 里 composition 而不动骨架
        let raw = std::fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(raw.trim()).unwrap();
        let mut tampered = v.clone();
        tampered["composition"]["total_tokens"] = serde_json::json!(999_999);
        std::fs::write(&path, serde_json::to_string(&tampered).unwrap() + "\n").unwrap();
        let report = verify_wal_file(&path).unwrap();
        assert!(!report.ok, "全快照覆盖下 composition 篡改必须破链");
        assert_eq!(report.first_bad_seq, Some(1));
        // 双保险锚点仍在：骨架伪造由 skeleton_hash 锚点 + 全量哈希双重拦截
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn hot_window_eviction_keeps_chain_and_prunes_index() {
        let path = temp_wal("eviction");
        let log = InspectorLog::open(&path).unwrap();
        let n = HOT_WINDOW_DEFAULT as u64 + 5;
        for i in 1..=n {
            let session = if i == 1 { Some(42) } else { None };
            log.append(InspectorRecord::Snapshot(snapshot(i, session, b"m")))
                .unwrap();
        }
        assert_eq!(log.total_records().unwrap(), n);
        // 热窗口只留 K 条；最早 session 42 的快照已被逐出
        assert!(
            log.snapshot_ids_of_session(42).unwrap().is_empty(),
            "逐出后索引修剪"
        );
        let report = log.verify_chain().unwrap();
        assert!(report.ok, "WAL 全量校验不受热窗口影响");
        assert_eq!(report.records, n);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn in_memory_mode_works_and_verifies() {
        let log = InspectorLog::in_memory();
        let s = snapshot(1, Some(3), b"m");
        log.append(InspectorRecord::Snapshot(s.clone())).unwrap();
        log.append(InspectorRecord::Observation(observation(&s, 5)))
            .unwrap();
        assert_eq!(log.total_records().unwrap(), 2);
        assert_eq!(log.snapshot_ids_of_session(3).unwrap(), vec![s.snapshot_id]);
        let report = log.verify_chain().unwrap();
        assert!(report.ok);
        // observation 哈希 = canonical 全量（同观测同哈希、异观测异哈希）
        let o = observation(&s, 5);
        let h1 = InspectorRecord::Observation(o.clone())
            .record_hash()
            .unwrap();
        let h2 = InspectorRecord::Observation(o).record_hash().unwrap();
        assert_eq!(h1, h2);
        let h3 = InspectorRecord::Observation(observation(&s, 6))
            .record_hash()
            .unwrap();
        assert_ne!(h1, h3);
    }

    #[test]
    fn first_record_chains_from_genesis() {
        let path = temp_wal("genesis");
        let log = InspectorLog::open(&path).unwrap();
        let s = snapshot(1, None, b"first");
        let record_hash = InspectorRecord::Snapshot(s.clone()).record_hash().unwrap();
        log.append(InspectorRecord::Snapshot(s)).unwrap();
        let expected_head = evorule_reactor::chain_step(GENESIS, &record_hash);
        assert_eq!(
            log.anchor_head().unwrap(),
            expected_head,
            "首记录链步自 genesis（验收 #11 链接语义）"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_existing_file_recovers_instead_of_truncating() {
        let path = temp_wal("open_recover");
        {
            let log = InspectorLog::open(&path).unwrap();
            log.append(InspectorRecord::Snapshot(snapshot(1, None, b"a")))
                .unwrap();
        }
        let log = InspectorLog::open(&path).unwrap(); // 二次 open = recover
        assert_eq!(log.total_records().unwrap(), 1);
        assert!(log.verify_chain().unwrap().ok);
        let _ = std::fs::remove_file(&path);
    }
}
