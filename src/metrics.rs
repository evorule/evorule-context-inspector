//! 健康指标 —— 五族全部落地：压力（§3.1，验收 #5）/ cache（§3.2）/
//! 漂移 + 三显式检测（§3.3，验收 #6 #7）/ 截断（§3.4，验收 #9）/
//! 重复率（§3.5，验收 #8）。
//!
//! 确定性纪律（§2.7）：比率一律以**万分比整数**产出（`*_permyriad`），
//! 浮点只在调用方展示层派生；所有输入路径无 panic（分母为 0 → 0）；
//! 压力分档用 u128 交叉相乘，不经浮点比较。

use crate::model::{ContextSkeleton, ContextSnapshot, UsageInfo};
use crate::ProviderKind;
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

/// 重复率窗口默认值：recent = 最近 5 条消息（§3.5，可配）。
pub const DEFAULT_RECENT_N: usize = 5;
/// shingle 词组长度默认值：连续 8 词（§3.5，可配）。
pub const DEFAULT_SHINGLE_K: usize = 8;

/// 默认截断标记正则：响应末尾 `truncated` 类标记（§3.4 通用路径，可配）。
/// 字符类含 `]`/`)`，覆盖 `[truncated]`、`(truncated)` 形态。
pub const DEFAULT_TRUNCATION_MARKER: &str = r"(?i)\btruncated[\s\].)!]*$";

static DEFAULT_MARKER: LazyLock<Result<Regex, String>> =
    LazyLock::new(|| Regex::new(DEFAULT_TRUNCATION_MARKER).map_err(|e| e.to_string()));

/// 词组集合：`text` 按空白切词、小写归一后，取连续 `k` 个词组成的词组集合。
///
/// 词数不足 `k` → 空集合（无 shingle；重复率分母为 0 时按 0 处理 — 验收 #8）。
fn shingles(text: &str, k: usize) -> std::collections::HashSet<String> {
    let words: Vec<String> = text.split_whitespace().map(str::to_lowercase).collect();
    let mut set = std::collections::HashSet::new();
    if k == 0 || words.len() < k {
        return set;
    }
    for window in words.windows(k) {
        set.insert(window.join(" "));
    }
    set
}

/// 重复率（万分比）：本快照内，最近 `recent_n` 条消息的 shingle 有多少比例
/// 也在更早消息里出现过（§3.5；口径统一为**本快照内**）。
///
/// 无 panic 路径：recent 为空、shingle 不足 k 个、earlier 为空 → 0。
/// 阈值口径：> 0.5（5000）Warning；> 0.8（8000）Critical（阈值判定在读取层）。
pub fn duplicate_ratio_permyriad(messages: &[String], recent_n: usize, shingle_k: usize) -> u32 {
    if messages.len() < 2 || recent_n == 0 {
        return 0;
    }
    let split_at = messages.len().saturating_sub(recent_n);
    let earlier = &messages[..split_at];
    let recent = &messages[split_at..];
    if earlier.is_empty() || recent.is_empty() {
        return 0;
    }
    let mut recent_shingles = std::collections::HashSet::new();
    for m in recent {
        recent_shingles.extend(shingles(m, shingle_k));
    }
    if recent_shingles.is_empty() {
        return 0; // 分母为 0 → 0（recent 不足 8 词等）
    }
    let mut earlier_shingles = std::collections::HashSet::new();
    for m in earlier {
        earlier_shingles.extend(shingles(m, shingle_k));
    }
    let overlap = recent_shingles.intersection(&earlier_shingles).count();
    // 分母 = |shingles(recent)|，已保证非 0；u32 内不溢出（交集 ≤ 分母 ≤ 集合规模）
    ((overlap as u128 * 10_000) / recent_shingles.len() as u128) as u32
}

/// 默认参数版（N=5、k=8）。
pub fn duplicate_ratio_permyriad_default(messages: &[String]) -> u32 {
    duplicate_ratio_permyriad(messages, DEFAULT_RECENT_N, DEFAULT_SHINGLE_K)
}

/// 截断检测（验收 #9）：finish_reason 路径 + 响应末尾标记正则，任一命中即截断。
///
/// - OpenAI / DeepSeek / Ollama：`finish_reason/done_reason == "length"`；
/// - Anthropic：`stop_reason == "max_tokens"`；
/// - 通用：`content_tail` 末尾匹配标记（默认 `DEFAULT_TRUNCATION_MARKER`，
///   可传自定义正则覆盖）。
pub fn detect_truncation(
    provider: ProviderKind,
    finish_reason: Option<&str>,
    content_tail: Option<&str>,
    custom_marker: Option<&Regex>,
) -> bool {
    let reason_hit = match provider {
        ProviderKind::Anthropic => finish_reason == Some("max_tokens"),
        _ => finish_reason == Some("length"),
    };
    if reason_hit {
        return true;
    }
    let Some(tail) = content_tail else {
        return false;
    };
    match custom_marker {
        Some(re) => re.is_match(tail),
        // 常量正则编译失败不可达；Err → 不命中（fail-safe，不误报截断）
        None => DEFAULT_MARKER
            .as_ref()
            .map(|re| re.is_match(tail))
            .unwrap_or(false),
    }
}

// ============ 压力（§3.1，验收 #5）============

/// 压力四档（左闭右开；恰等归高档：0.6→Warning、0.8→Critical、0.95→Overflow）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureBand {
    /// [0, 0.6)
    Healthy,
    /// [0.6, 0.8)
    Warning,
    /// [0.8, 0.95)
    Critical,
    /// [0.95, ∞)
    Overflow,
}

/// 压力档位判定（整数运算：total×10000 与 window×阈值 交叉相乘，u128 防溢出，
/// 不经浮点——跨平台字节一致的确定性红线）。
pub fn pressure_band(total_tokens: u64, context_window: u32) -> PressureBand {
    if context_window == 0 {
        // 窗口未知（§2.6 Custom 未声明）→ pressure 无意义；此处按最坏档兜底
        return PressureBand::Overflow;
    }
    let lhs = u128::from(total_tokens) * 10_000;
    let w = u128::from(context_window);
    if lhs < w * 6_000 {
        PressureBand::Healthy
    } else if lhs < w * 8_000 {
        PressureBand::Warning
    } else if lhs < w * 9_500 {
        PressureBand::Critical
    } else {
        PressureBand::Overflow
    }
}

// ============ cache 命中率（§3.2）============

/// cache 命中率（万分比）：cached/input。provider 未回报（Ollama 等）或
/// input 为 None/0 → None（指标不存在，不虚构 0）。
pub fn cache_hit_permyriad(usage: &UsageInfo) -> Option<u32> {
    let cached = u128::from(usage.cached_tokens?);
    let input = usage.input_tokens?;
    if input == 0 {
        return None;
    }
    Some(((cached * 10_000) / u128::from(input)) as u32)
}

/// cache 档位：<1000（0.1）Critical；<3000（0.3）Warning；其余 Healthy。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheBand {
    Healthy,
    Warning,
    Critical,
}

pub fn cache_hit_band(hit_permyriad: u32) -> CacheBand {
    if hit_permyriad < 1_000 {
        CacheBand::Critical
    } else if hit_permyriad < 3_000 {
        CacheBand::Warning
    } else {
        CacheBand::Healthy
    }
}

// ============ 漂移（§3.3，验收 #6 #7）============

/// 结构指纹 = multiset{消息 content_hash} ∪ multiset{工具 schema_hash}。
/// 计数信息保留（同内容 3 条贡献 3 元素）；**system_hash 刻意不在指纹内**
/// ——system 变更走显式检测（SystemPromptChanged），漂移专测对话/工具面演进。
fn fingerprint(skeleton: &ContextSkeleton) -> BTreeMap<&str, u64> {
    let mut m: BTreeMap<&str, u64> = BTreeMap::new();
    for msg in &skeleton.messages {
        *m.entry(msg.content_hash.as_str()).or_default() += 1;
    }
    for tool in &skeleton.tools {
        *m.entry(tool.schema_hash.as_str()).or_default() += 1;
    }
    m
}

/// drift = multiset Jaccard 距离 = 1 - |∩|/|∪|，万分比（drift_permyriad）。
/// 双方全空（∪=0）→ 0（无漂移语义而非满漂移）。
pub fn drift_permyriad(prev: &ContextSkeleton, curr: &ContextSkeleton) -> u32 {
    let a = fingerprint(prev);
    let b = fingerprint(curr);
    let mut inter: u128 = 0;
    let mut union: u128 = 0;
    for (key, a_count) in &a {
        let b_count = b.get(key).copied().unwrap_or(0);
        inter += u128::from((*a_count).min(b_count));
        union += u128::from((*a_count).max(b_count));
    }
    for (key, b_count) in &b {
        if !a.contains_key(key) {
            union += u128::from(*b_count);
        }
    }
    if union == 0 {
        return 0;
    }
    (((union - inter) * 10_000) / union) as u32
}

/// 漂移档位：>9000（0.9）Critical；>7000（0.7）Warning；其余 Ok。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftBand {
    Ok,
    Warning,
    Critical,
}

pub fn drift_band(drift_permyriad: u32) -> DriftBand {
    if drift_permyriad > 9_000 {
        DriftBand::Critical
    } else if drift_permyriad > 7_000 {
        DriftBand::Warning
    } else {
        DriftBand::Ok
    }
}

/// 三个显式检测事件（独立于 drift_score，直接比对上一快照，§3.3）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextEvent {
    /// system_hash ≠（Critical）
    SystemPromptChanged,
    /// 工具名集合对称差非空（Warning）
    ToolSetChanged,
    /// message_count 或 total_tokens 降 ≥50%（Warning）
    ContextReset,
}

/// 三显式检测（验收 #6/#7 的判定层）。
pub fn detect_context_events(prev: &ContextSnapshot, curr: &ContextSnapshot) -> Vec<ContextEvent> {
    let mut out = Vec::new();
    if prev.skeleton.system_hash != curr.skeleton.system_hash {
        out.push(ContextEvent::SystemPromptChanged);
    }
    let prev_tools: BTreeSet<&str> = prev
        .skeleton
        .tools
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    let curr_tools: BTreeSet<&str> = curr
        .skeleton
        .tools
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    if prev_tools
        .symmetric_difference(&curr_tools)
        .next()
        .is_some()
    {
        out.push(ContextEvent::ToolSetChanged);
    }
    let pm = prev.composition.message_count;
    let cm = curr.composition.message_count;
    let pt = prev.composition.total_tokens;
    let ct = curr.composition.total_tokens;
    let msg_reset = pm > 0 && cm * 2 <= pm;
    let tok_reset = pt > 0 && ct * 2 <= pt;
    if msg_reset || tok_reset {
        out.push(ContextEvent::ContextReset);
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    fn m(s: &str) -> String {
        s.to_owned()
    }

    // ---- 验收 #8：word-shingle（对照手算）----

    #[test]
    fn duplicate_full_overlap_is_permyriad_10k() {
        // earlier 一条 8 词消息，recent 同内容 → 交集/分母 = 1/1 → 10000
        let msgs = vec![m("a b c d e f g h"), m("A B C D E F G H")];
        assert_eq!(
            duplicate_ratio_permyriad(&msgs, 1, 8),
            10_000,
            "大小写归一后应全重叠"
        );
    }

    #[test]
    fn duplicate_partial_overlap_hand_computed() {
        // earlier: "a..h" → shingle {abcdefgh}
        // recent:  "a..j" → shingles {abcdefgh, bcdefghi, cdefghij} 共 3，交集 1
        // → 1/3 = 3333.33 → 3333
        let msgs = vec![m("a b c d e f g h"), m("a b c d e f g h i j")];
        assert_eq!(duplicate_ratio_permyriad(&msgs, 1, 8), 3_333);
    }

    #[test]
    fn duplicate_no_overlap_is_zero() {
        let msgs = vec![m("a b c d e f g h"), m("x y z w q r s t")];
        assert_eq!(duplicate_ratio_permyriad(&msgs, 1, 8), 0);
    }

    #[test]
    fn duplicate_short_recent_denominator_zero_yields_zero_no_panic() {
        // recent 只有 3 词（< k=8）→ 无 shingle → 分母 0 → 0（无 panic 路径）
        let msgs = vec![m("a b c d e f g h"), m("a b c")];
        assert_eq!(duplicate_ratio_permyriad(&msgs, 1, 8), 0);
    }

    #[test]
    fn duplicate_degenerate_inputs_yield_zero() {
        assert_eq!(duplicate_ratio_permyriad(&[], 5, 8), 0, "空消息");
        assert_eq!(
            duplicate_ratio_permyriad(&[m("only one message")], 5, 8),
            0,
            "单条无 earlier"
        );
        assert_eq!(
            duplicate_ratio_permyriad(&[m("a"), m("b")], 0, 8),
            0,
            "recent_n=0"
        );
        assert_eq!(
            duplicate_ratio_permyriad(&[m("a"), m("b")], 5, 0),
            0,
            "k=0 防御"
        );
    }

    #[test]
    fn duplicate_recent_window_takes_last_n() {
        // 7 条，N=5：earlier = 前 2 条 [base, novelA]，recent = 后 5 条 [base, n1..n4]
        // base shingle 双侧共有 → 交集 1；recent shingle 集 = {base, n1..n4} = 5
        // → 1/5 = 2000（手算：交集 ≤ 分母，filler 不参与）
        let msgs = vec![
            m("a b c d e f g h"),
            m("i j k l m n o p"),
            m("a b c d e f g h"),
            m("q1 q2 q3 q4 q5 q6 q7 q8"),
            m("r1 r2 r3 r4 r5 r6 r7 r8"),
            m("s1 s2 s3 s4 s5 s6 s7 s8"),
            m("t1 t2 t3 t4 t5 t6 t7 t8"),
        ];
        assert_eq!(duplicate_ratio_permyriad_default(&msgs), 2_000);
    }

    // ---- 验收 #9：截断检测（参数化）----

    #[test]
    fn truncation_by_finish_reason_per_provider() {
        assert!(detect_truncation(
            ProviderKind::OpenAi,
            Some("length"),
            None,
            None
        ));
        assert!(detect_truncation(
            ProviderKind::DeepSeek,
            Some("length"),
            None,
            None
        ));
        assert!(detect_truncation(
            ProviderKind::Anthropic,
            Some("max_tokens"),
            None,
            None
        ));
        assert!(detect_truncation(
            ProviderKind::Ollama,
            Some("length"),
            None,
            None
        ));
        assert!(!detect_truncation(
            ProviderKind::OpenAi,
            Some("stop"),
            None,
            None
        ));
        assert!(!detect_truncation(
            ProviderKind::Anthropic,
            Some("end_turn"),
            None,
            None
        ));
        assert!(!detect_truncation(
            ProviderKind::OpenAi,
            Some("tool_calls"),
            None,
            None
        ));
        assert!(!detect_truncation(ProviderKind::Ollama, None, None, None));
    }

    #[test]
    fn truncation_by_content_tail_marker() {
        assert!(detect_truncation(
            ProviderKind::OpenAi,
            Some("stop"),
            Some("...and then [truncated]"),
            None
        ));
        assert!(detect_truncation(
            ProviderKind::Anthropic,
            Some("end_turn"),
            Some("output Truncated."),
            None
        ));
        assert!(!detect_truncation(
            ProviderKind::OpenAi,
            Some("stop"),
            Some("all good, nothing cut"),
            None
        ));
    }

    #[test]
    fn truncation_custom_marker_regex_overrides_default() {
        let custom = Regex::new(r"##CUT##$").unwrap();
        assert!(detect_truncation(
            ProviderKind::OpenAi,
            Some("stop"),
            Some("partial ##CUT##"),
            Some(&custom)
        ));
        // 自定义正则下默认标记不再命中
        assert!(!detect_truncation(
            ProviderKind::OpenAi,
            Some("stop"),
            Some("end [truncated]"),
            Some(&custom)
        ));
    }

    // ---- 验收 #5：压力四档（含 0.6/0.8/0.95 恰等边界）----

    #[test]
    fn pressure_bands_with_exact_boundaries() {
        let w: u32 = 1_000;
        assert_eq!(pressure_band(0, w), PressureBand::Healthy);
        assert_eq!(pressure_band(599, w), PressureBand::Healthy);
        // 恰等 0.6 → Warning（左闭右开，恰等归高档）
        assert_eq!(pressure_band(600, w), PressureBand::Warning);
        assert_eq!(pressure_band(799, w), PressureBand::Warning);
        assert_eq!(pressure_band(800, w), PressureBand::Critical);
        assert_eq!(pressure_band(949, w), PressureBand::Critical);
        assert_eq!(pressure_band(950, w), PressureBand::Overflow);
        assert_eq!(pressure_band(1_500, w), PressureBand::Overflow);
        // 整数交叉相乘下非整除边界：31/50 = 0.62 → Warning；30/50 = 0.6 恰等 → Warning
        assert_eq!(pressure_band(31, 50), PressureBand::Warning);
        assert_eq!(pressure_band(30, 50), PressureBand::Warning);
        assert_eq!(pressure_band(29, 50), PressureBand::Healthy);
    }

    #[test]
    fn pressure_zero_window_falls_to_overflow() {
        assert_eq!(pressure_band(1, 0), PressureBand::Overflow);
        assert_eq!(pressure_band(0, 0), PressureBand::Overflow);
    }

    #[test]
    fn pressure_exact_ratios_u64_scale() {
        // 大数交叉相乘不溢出：u64::MAX/2 与 1M 窗口
        let big = u64::MAX / 2;
        assert_eq!(pressure_band(big, 1_000_000), PressureBand::Overflow);
    }

    // ---- cache 命中率 ----

    #[test]
    fn cache_hit_permyriad_and_bands() {
        let u = |c: Option<u64>, i: Option<u64>| UsageInfo {
            input_tokens: i,
            output_tokens: None,
            cached_tokens: c,
        };
        assert_eq!(cache_hit_permyriad(&u(Some(300), Some(1000))), Some(3_000));
        assert_eq!(cache_hit_band(3_000), CacheBand::Healthy);
        assert_eq!(cache_hit_band(2_999), CacheBand::Warning);
        assert_eq!(cache_hit_band(1_000), CacheBand::Warning);
        assert_eq!(cache_hit_band(999), CacheBand::Critical);
        assert_eq!(
            cache_hit_permyriad(&u(None, Some(1000))),
            None,
            "未回报 cached → None"
        );
        assert_eq!(
            cache_hit_permyriad(&u(Some(5), None)),
            None,
            "未回报 input → None"
        );
        assert_eq!(
            cache_hit_permyriad(&u(Some(5), Some(0))),
            None,
            "input=0 → None"
        );
    }

    // ---- 验收 #6：换 system 原文 → drift=0 但 SystemPromptChanged 必触发 ----

    fn snap_with(system: &[u8], messages: &[&[u8]], tools: &[&[u8]]) -> ContextSnapshot {
        use crate::model::{Composition, ContextSkeleton, MessageEntry, SessionLink, ToolEntry};
        use crate::TokenSource;
        let skeleton = ContextSkeleton::new(
            crate::ProviderKind::DeepSeek,
            "deepseek-chat",
            system,
            tools
                .iter()
                .enumerate()
                .map(|(i, s)| ToolEntry::new(format!("t{i}"), s))
                .collect(),
            messages
                .iter()
                .map(|s| MessageEntry::new("user", crate::ContentKind::Text, s, 0))
                .collect(),
        );
        let composition = Composition::new(
            10,
            10,
            100,
            0,
            messages.len() as u32,
            tools.len() as u32,
            TokenSource::Estimated,
        )
        .unwrap_or_else(|e| panic!("composition: {e}"));
        ContextSnapshot::new(
            uuid::Uuid::new_v4(),
            SessionLink::new(None, crate::BindingKind::Unattributed, None),
            0,
            crate::ProviderKind::DeepSeek,
            "deepseek-chat",
            None,
            composition,
            skeleton,
        )
        .unwrap_or_else(|e| panic!("snapshot: {e}"))
    }

    #[test]
    fn system_swap_keeps_drift_zero_but_fires_explicit_detection() {
        let prev = snap_with(b"system v1", &[b"m1", b"m2"], &[b"tool-a"]);
        let curr = snap_with(b"system v2 INJECTED", &[b"m1", b"m2"], &[b"tool-a"]);
        assert_eq!(
            drift_permyriad(&prev.skeleton, &curr.skeleton),
            0,
            "同 role/kind 换 system → 指纹不变"
        );
        assert_eq!(drift_band(0), DriftBand::Ok);
        let events = detect_context_events(&prev, &curr);
        assert_eq!(
            events,
            vec![ContextEvent::SystemPromptChanged],
            "显式检测必须触发（验收 #6 命门）"
        );
    }

    // ---- 验收 #7：multiset Jaccard 手算 + ToolSetChanged + ContextReset ----

    #[test]
    fn multiset_jaccard_counts_multiplicity() {
        // prev 消息 [x]×1；curr 消息 [x]×3 → ∩=1 ∪=3 → drift = 2/3 = 6666
        let prev = snap_with(b"s", &[b"x"], &[]);
        let curr = snap_with(b"s", &[b"x", b"x", b"x"], &[]);
        assert_eq!(drift_permyriad(&prev.skeleton, &curr.skeleton), 6_666);
        // 同内容 3 条对 3 条 → drift 0（计数保留语义）
        let curr3 = snap_with(b"s", &[b"x", b"x", b"x"], &[]);
        let prev3 = snap_with(b"s", &[b"x", b"x", b"x"], &[]);
        assert_eq!(drift_permyriad(&prev3.skeleton, &curr3.skeleton), 0);
        // 完全不同 → 10000
        let other = snap_with(b"s", &[b"y", b"z"], &[]);
        assert_eq!(drift_permyriad(&prev.skeleton, &other.skeleton), 10_000);
        // 双空 → 0
        let e1 = snap_with(b"s", &[], &[]);
        let e2 = snap_with(b"s", &[], &[]);
        assert_eq!(drift_permyriad(&e1.skeleton, &e2.skeleton), 0);
    }

    #[test]
    fn tool_set_change_fires_on_name_symmetric_difference() {
        let prev = snap_with(b"s", &[b"m"], &[b"tool-a"]);
        let curr = snap_with(b"s", &[b"m"], &[b"tool-a"]);
        assert!(
            detect_context_events(&prev, &curr).is_empty(),
            "无变化 → 无事件"
        );
        // 同名换 schema：名字对称差为空 → 不触发 ToolSetChanged；但进漂移指纹
        let rehashed = snap_with(b"s", &[b"m"], &[b"tool-b"]);
        assert!(detect_context_events(&prev, &rehashed).is_empty());
        assert!(drift_permyriad(&prev.skeleton, &rehashed.skeleton) > 0);
        // 名字集合变化（删/增）→ 触发
        let removed = snap_with(b"s", &[b"m"], &[]);
        assert_eq!(
            detect_context_events(&prev, &removed),
            vec![ContextEvent::ToolSetChanged]
        );
        let added = snap_with(b"s", &[b"m"], &[b"tool-a", b"tool-b"]);
        assert_eq!(
            detect_context_events(&prev, &added),
            vec![ContextEvent::ToolSetChanged]
        );
    }

    #[test]
    fn context_reset_fires_at_half_drop() {
        // 消息 4 → 2（恰 50%）→ ContextReset
        let prev = snap_with(b"s", &[b"m1", b"m2", b"m3", b"m4"], &[]);
        let curr = snap_with(b"s", &[b"m1", b"m2"], &[]);
        assert_eq!(
            detect_context_events(&prev, &curr),
            vec![ContextEvent::ContextReset]
        );
        // token 判据：消息数不变（1→1），total 120→60 恰减半 → 触发
        let prev_tok = snap_with(b"s", &[b"m1"], &[]);
        let mut curr_tok = snap_with(b"s", &[b"m1"], &[]);
        curr_tok.composition.total_tokens /= 2;
        assert_eq!(
            detect_context_events(&prev_tok, &curr_tok),
            vec![ContextEvent::ContextReset]
        );
        // 降幅不足 50% → 不触发
        let mild = snap_with(b"s", &[b"m1", b"m2", b"m3"], &[]);
        assert!(detect_context_events(&prev, &mild).is_empty());
        // prev 为 0 → 不触发（防 0 语义误报）
        let zero = snap_with(b"s", &[], &[]);
        assert!(detect_context_events(&zero, &zero).is_empty());
    }
}
