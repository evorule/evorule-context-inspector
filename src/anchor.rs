//! 插件自持 G-A1 锚点流（实施文档决策 11，v3.3 修订：**不动 evorule-cli**，
//! 项目方 2026-10-02 指令「不动 evorule-cli，verify 在 server 侧/插件侧解决」）。
//!
//! - **格式复用**（决策 11 保留的部分）：anchor_payload 五字段
//!   （seq/version/entry_count/last_hash/prev_anchor_hash，serde_json 字典序）
//!   与自哈希公式 `blake3(载荷+签名)` 与 cli `verify_anchors.rs` 同款——
//!   离线审计者用同一套公式即可核验，无需 cli 扩展。
//! - **签名原语复用**：`evorule_governance::signing`（Ed25519，G-A1 同源），
//!   禁自写密码学。
//! - **自持**：密钥与锚点流均在本插件 data 目录（server 永不读写，契约 §2.1）；
//!   每条锚点内嵌 verifying_key（自包含，可离线验）。
//! - 发布节律：每 N 条（默认 100，`--anchor-every-n` 可配）。

use crate::chain::{InspectorLog, GENESIS};
use evorule_governance::signing::AuditSigner;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// 锚点载荷版本字段（对齐 cli anchor_payload 的 version 语义）。
pub const ANCHOR_VERSION: u64 = 1;
/// 默认发布节律：每 100 条一锚。
pub const ANCHOR_EVERY_N_DEFAULT: u64 = 100;
/// 密钥文件名（data 目录内）。
pub const KEY_FILE: &str = "anchor-key.json";

/// 单条锚点（自包含 verifying_key，可离线核验）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorEntry {
    pub seq: u64,
    pub version: u64,
    pub entry_count: u64,
    pub last_hash: String,
    pub prev_anchor_hash: String,
    /// hex（Ed25519 签名，64 字节）
    pub signature: String,
    /// hex（32 字节验证公钥；每条内嵌 = 流自包含）
    pub verifying_key: String,
}

/// 载荷重算（与 cli `anchor_payload` 同构：json! 五字段，BTreeMap 字典序）。
pub fn anchor_payload(
    seq: u64,
    version: u64,
    entry_count: u64,
    last_hash: &str,
    prev_anchor_hash: &str,
) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "seq": seq,
        "version": version,
        "entry_count": entry_count,
        "last_hash": last_hash,
        "prev_anchor_hash": prev_anchor_hash,
    }))
    .unwrap_or_default()
}

/// 锚点自哈希 = blake3(载荷 + 签名)（与 cli 同公式，供下一条链接）。
pub fn anchor_self_hash(payload: &[u8], signature: &[u8; 64]) -> String {
    let mut preimage = payload.to_vec();
    preimage.extend_from_slice(signature);
    blake3_hash_hex(&preimage)
}

// B 族：禁自写 blake3——此处复用 evorule-hash 的 digest 原语。
fn blake3_hash_hex(bytes: &[u8]) -> String {
    evorule_hash::digest(bytes)
}

/// 锚点服务：发布（节律触发）+ 全流验证。
pub struct AnchorService {
    chain: Arc<InspectorLog>,
    anchors_path: PathBuf,
    signer: AuditSigner,
    every_n: u64,
}

/// 验证报告。
#[derive(Debug, Clone, serde::Serialize)]
pub struct AnchorReport {
    pub entries: u64,
    pub ok: bool,
    pub first_bad_seq: Option<u64>,
    pub reason: Option<String>,
    /// 末锚与当前链头/条数的一致性（ok 的一部分）。
    pub covers_chain: bool,
}

impl AnchorService {
    /// 从 data 目录装载（密钥缺失则生成并落盘；0600 由调用方目录权限保障）。
    pub fn load_or_create(
        data_dir: &Path,
        chain: Arc<InspectorLog>,
        every_n: u64,
    ) -> Result<Self, crate::InspectorError> {
        std::fs::create_dir_all(data_dir)
            .map_err(|e| crate::InspectorError::Io(format!("锚点目录创建失败: {e}")))?;
        let key_path = data_dir.join(KEY_FILE);
        let signer = if key_path.exists() {
            let raw = std::fs::read_to_string(&key_path)
                .map_err(|e| crate::InspectorError::Io(format!("密钥读取失败: {e}")))?;
            let v: serde_json::Value = serde_json::from_str(&raw)
                .map_err(|e| crate::InspectorError::Io(format!("密钥解析失败: {e}")))?;
            let seed = v
                .get("seed_hex")
                .and_then(|s| s.as_str())
                .ok_or_else(|| crate::InspectorError::Io("密钥文件缺 seed_hex".into()))?;
            AuditSigner::from_hex(seed)
                .map_err(|e| crate::InspectorError::Io(format!("密钥装载失败: {e}")))?
        } else {
            let (seed_hex, _vk) = AuditSigner::generate_keys()
                .map_err(|e| crate::InspectorError::Io(format!("密钥生成失败: {e}")))?;
            let body = serde_json::json!({
                "seed_hex": seed_hex,
                "note": "G-A1 插件自持锚点密钥（Ed25519 seed）——泄露=锚点可被伪造，妥存"
            });
            std::fs::write(
                &key_path,
                serde_json::to_string_pretty(&body).unwrap_or_default(),
            )
            .map_err(|e| crate::InspectorError::Io(format!("密钥写入失败: {e}")))?;
            AuditSigner::from_hex(&seed_hex)
                .map_err(|e| crate::InspectorError::Io(format!("密钥装载失败: {e}")))?
        };
        Ok(Self {
            chain,
            anchors_path: data_dir.join("anchors.jsonl"),
            signer,
            every_n: every_n.max(1),
        })
    }

    /// 节律触发：距上一锚点 ≥ N 条 → 发布新锚点（幂等：不足则 None）。
    pub fn publish_if_due(&self) -> Result<Option<AnchorEntry>, crate::InspectorError> {
        let total = self.chain.total_records()?;
        let last = self.last_entry()?;
        let anchored = last.as_ref().map(|e| e.entry_count).unwrap_or(0);
        if total.saturating_sub(anchored) < self.every_n {
            return Ok(None);
        }
        let head = self.chain.anchor_head()?;
        let seq = last.as_ref().map(|e| e.seq + 1).unwrap_or(1);
        let prev_anchor_hash = match &last {
            Some(e) => {
                let sig = decode_sig(&e.signature).map_err(|reason| {
                    crate::InspectorError::Io(format!("上一锚点签名解析失败: {reason}"))
                })?;
                anchor_self_hash(
                    &anchor_payload(
                        e.seq,
                        e.version,
                        e.entry_count,
                        &e.last_hash,
                        &e.prev_anchor_hash,
                    ),
                    &sig,
                )
            }
            None => GENESIS.to_owned(),
        };
        let payload = anchor_payload(seq, ANCHOR_VERSION, total, &head, &prev_anchor_hash);
        let signature = self.signer.signature_bytes(&payload);
        let entry = AnchorEntry {
            seq,
            version: ANCHOR_VERSION,
            entry_count: total,
            last_hash: head,
            prev_anchor_hash,
            signature: hex_encode(&signature),
            verifying_key: self.signer.verifying_hex(),
        };
        let line = serde_json::to_string(&entry)
            .map_err(|e| crate::InspectorError::Io(format!("锚点序列化失败: {e}")))?;
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.anchors_path)
            .map_err(|e| crate::InspectorError::Io(format!("锚点流打开失败: {e}")))?;
        writeln!(f, "{line}")
            .map_err(|e| crate::InspectorError::Io(format!("锚点写入失败: {e}")))?;
        f.flush()
            .map_err(|e| crate::InspectorError::Io(format!("锚点 flush 失败: {e}")))?;
        Ok(Some(entry))
    }

    fn last_entry(&self) -> Result<Option<AnchorEntry>, crate::InspectorError> {
        let entries = self.read_entries()?;
        Ok(entries.last().cloned())
    }

    fn read_entries(&self) -> Result<Vec<AnchorEntry>, crate::InspectorError> {
        let path = &self.anchors_path;
        if !path.exists() {
            return Ok(Vec::new());
        }
        let raw = std::fs::read_to_string(path)
            .map_err(|e| crate::InspectorError::Io(format!("锚点流读取失败: {e}")))?;
        let mut out = Vec::new();
        for (i, line) in raw.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            out.push(serde_json::from_str(line).map_err(|e| {
                crate::InspectorError::WalCorrupt {
                    seq: i as u64 + 1,
                    reason: format!("锚点行解析失败: {e}"),
                }
            })?);
        }
        Ok(out)
    }

    /// 全流验证：逐条（载荷重算→验签→自哈希链接）+ 末锚覆盖链头/条数。
    pub fn verify(&self) -> Result<AnchorReport, crate::InspectorError> {
        let entries = self.read_entries()?;
        let mut prev_self_hash = GENESIS.to_owned();
        for (idx, e) in entries.iter().enumerate() {
            let seq = e.seq;
            if e.seq != idx as u64 + 1 {
                return Ok(fail(
                    entries.len() as u64,
                    seq,
                    format!("seq 不连续：期望 {}，实得 {}", idx + 1, e.seq),
                ));
            }
            if e.prev_anchor_hash != prev_self_hash {
                return Ok(fail(
                    entries.len() as u64,
                    seq,
                    format!("prev_anchor_hash 不接续：期望 {prev_self_hash}"),
                ));
            }
            let (sig, vk, payload) = match (decode_sig(&e.signature), decode_key(&e.verifying_key))
            {
                (Ok(sig), Ok(vk)) => (
                    sig,
                    vk,
                    anchor_payload(
                        e.seq,
                        e.version,
                        e.entry_count,
                        &e.last_hash,
                        &e.prev_anchor_hash,
                    ),
                ),
                (Err(reason), _) | (_, Err(reason)) => {
                    return Ok(fail(entries.len() as u64, seq, reason))
                }
            };
            let ok = match evorule_governance::signing::verify_signature(vk, &payload, &sig) {
                Ok(ok) => ok,
                Err(e) => return Ok(fail(entries.len() as u64, seq, format!("验签器错误: {e}"))),
            };
            if !ok {
                return Ok(fail(
                    entries.len() as u64,
                    seq,
                    "签名验签失败（防抵赖拦截）".into(),
                ));
            }
            prev_self_hash = anchor_self_hash(&payload, &sig);
        }
        // 末锚覆盖当前链
        let total = self.chain.total_records()?;
        let head = self.chain.anchor_head()?;
        let covers = match entries.last() {
            Some(e) => e.entry_count == total && e.last_hash == head,
            None => total == 0,
        };
        Ok(AnchorReport {
            entries: entries.len() as u64,
            ok: true,
            first_bad_seq: None,
            reason: None,
            covers_chain: covers,
        })
    }

    pub fn anchors_path(&self) -> &Path {
        &self.anchors_path
    }
}

fn fail(entries: u64, seq: u64, reason: String) -> AnchorReport {
    AnchorReport {
        entries,
        ok: false,
        first_bad_seq: Some(seq),
        reason: Some(reason),
        covers_chain: false,
    }
}

fn decode_sig(hex: &str) -> Result<[u8; 64], String> {
    let bytes = hex_decode(hex).ok_or_else(|| "签名 hex 非法".to_owned())?;
    bytes
        .try_into()
        .map_err(|_| "签名长度非 64 字节".to_owned())
}

fn decode_key(hex: &str) -> Result<[u8; 32], String> {
    let bytes = hex_decode(hex).ok_or_else(|| "公钥 hex 非法".to_owned())?;
    bytes
        .try_into()
        .map_err(|_| "公钥长度非 32 字节".to_owned())
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok())
        .collect()
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::chain::{InspectorLog, InspectorRecord};
    use std::sync::Arc;

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "anchor_test_{}_{}",
            tag,
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn payload_is_lexicographic_canonical() {
        // 与 cli anchor_payload 字节同构（决策 11 格式复用）：BTreeMap 字典序
        let payload = anchor_payload(3, 1, 42, "deadbeef", "genesis");
        assert_eq!(
            String::from_utf8(payload).unwrap(),
            r#"{"entry_count":42,"last_hash":"deadbeef","prev_anchor_hash":"genesis","seq":3,"version":1}"#
        );
    }

    #[test]
    fn publish_on_cadence_and_verify_covers_chain() {
        let dir = temp_dir("cadence");
        let chain = Arc::new(InspectorLog::in_memory());
        let svc = AnchorService::load_or_create(&dir, chain.clone(), 2).unwrap();
        // 密钥文件生成
        assert!(dir.join(KEY_FILE).exists());
        assert!(
            !dir.join("anchors.jsonl").exists(),
            "尚无锚点时不产生流文件"
        );

        chain
            .append(InspectorRecord::Snapshot(test_snapshot(1)))
            .unwrap();
        assert!(svc.publish_if_due().unwrap().is_none(), "不足节律 → 不发布");

        chain
            .append(InspectorRecord::Snapshot(test_snapshot(2)))
            .unwrap();
        let e1 = svc.publish_if_due().unwrap().expect("达到节律 → 发布");
        assert_eq!((e1.seq, e1.entry_count), (1, 2));
        assert_ne!(e1.last_hash, GENESIS);

        chain
            .append(InspectorRecord::Snapshot(test_snapshot(3)))
            .unwrap();
        assert!(svc.publish_if_due().unwrap().is_none());
        chain
            .append(InspectorRecord::Snapshot(test_snapshot(4)))
            .unwrap();
        let e2 = svc.publish_if_due().unwrap().expect("第二锚");
        assert_eq!(e2.seq, 2);
        assert_eq!(e2.prev_anchor_hash.chars().count(), 64);

        let report = svc.verify().unwrap();
        assert!(report.ok, "reason={:?}", report.reason);
        assert!(report.covers_chain, "末锚必须覆盖当前链头/条数");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tampered_anchor_fails_verification() {
        let dir = temp_dir("tamper");
        let chain = Arc::new(InspectorLog::in_memory());
        let svc = AnchorService::load_or_create(&dir, chain.clone(), 1).unwrap();
        chain
            .append(InspectorRecord::Snapshot(test_snapshot(1)))
            .unwrap();
        svc.publish_if_due().unwrap().expect("每 1 条一锚");

        // 篡改 entry_count（载荷变 → 签名不再匹配）
        let path = dir.join("anchors.jsonl");
        let raw = std::fs::read_to_string(&path).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(raw.trim()).unwrap();
        v["entry_count"] = serde_json::json!(999);
        std::fs::write(&path, serde_json::to_string(&v).unwrap() + "\n").unwrap();

        let report = svc.verify().unwrap();
        assert!(!report.ok);
        assert_eq!(report.first_bad_seq, Some(1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn test_snapshot(seq: u64) -> crate::model::ContextSnapshot {
        use crate::model::{
            Composition, ContextSkeleton, ContextSnapshot, MessageEntry, SessionLink, ToolEntry,
        };
        use crate::TokenSource;
        let skeleton = ContextSkeleton::new(
            crate::ProviderKind::DeepSeek,
            "deepseek-chat",
            b"sys",
            vec![ToolEntry::new("t", b"schema")],
            vec![MessageEntry::new(
                "user",
                crate::ContentKind::Text,
                format!("m{seq}").as_bytes(),
                0,
            )],
        );
        let composition = Composition::new(3, 5, 20, 0, 1, 1, TokenSource::Estimated).unwrap();
        ContextSnapshot::new(
            uuid::Uuid::from_u128(u128::from(seq)),
            SessionLink::new(None, crate::BindingKind::Unattributed, None),
            seq * 1000,
            crate::ProviderKind::DeepSeek,
            "deepseek-chat",
            None,
            composition,
            skeleton,
        )
        .unwrap()
    }
}
