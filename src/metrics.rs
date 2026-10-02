//! 健康指标 —— Phase 2 交付：重复率（word-shingle 包含度，验收 #8）与
//! 截断检测（验收 #9）。压力 / cache / 漂移三族指标随 Phase 3 补齐。
//!
//! 确定性纪律（§2.7）：比率一律以**万分比整数**产出（`*_permyriad`），
//! 浮点只在调用方展示层派生；所有输入路径无 panic（分母为 0 → 0）。

use crate::ProviderKind;
use regex::Regex;
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
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
}
