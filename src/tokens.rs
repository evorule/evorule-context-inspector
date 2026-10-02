//! token 估算 —— 绑定 tiktoken（实施文档 v3：不自研分词）。
//!
//! - OpenAI 家族新模型走 `o200k_base`，其余走 `cl100k_base`；
//! - Anthropic / DeepSeek / Ollama 的原生分词器不公开，用 cl100k 词元数
//!   **近似**（`TokenSource::Estimated` 暂态，ProviderReported 回填后为准，
//!   偏差 >10% 出 `TokenEstimateDiverged` issue —— 指标层职责）。
//! - BPE 实例进程级缓存（构造含正则编译，代价高）；初始化失败映射为
//!   `InspectorError`，不 panic（G5）。

use crate::{InspectorError, ProviderKind};
use std::sync::LazyLock;
use tiktoken_rs::CoreBPE;

static CL100K: LazyLock<Result<CoreBPE, String>> =
    LazyLock::new(|| tiktoken_rs::cl100k_base().map_err(|e| e.to_string()));
static O200K: LazyLock<Result<CoreBPE, String>> =
    LazyLock::new(|| tiktoken_rs::o200k_base().map_err(|e| e.to_string()));

fn bpe(
    which: &'static LazyLock<Result<CoreBPE, String>>,
    name: &str,
) -> Result<&'static CoreBPE, InspectorError> {
    which
        .as_ref()
        .map_err(|e| InspectorError::InvalidInput(format!("tiktoken {name} 初始化失败：{e}")))
}

/// 模型 → 词元编码选择。OpenAI o 系/gpt-4o 系/4.1 系走 o200k，其余（含近似场景）cl100k。
fn encoding_for(model: &str) -> &'static LazyLock<Result<CoreBPE, String>> {
    let m = model.to_ascii_lowercase();
    if m.starts_with("gpt-4o")
        || m.starts_with("gpt-4.1")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
    {
        &O200K
    } else {
        &CL100K
    }
}

/// 估算一段文本的 token 数（近似口径，见模块注释）。
pub fn estimate_text(model: &str, text: &str) -> Result<u64, InspectorError> {
    let enc = bpe(encoding_for(model), "编码器")?;
    Ok(enc.encode_ordinary(text).len() as u64)
}

/// 估算某 provider 家族的图片 token（各家计法不同，设计稿 §4 count_image_tokens）。
///
/// - OpenAI（high detail，官方贴片公式）：短边缩到 ≤768、长边 ≤2048 后，
///   85 + 170 × ceil(w/512) × ceil(h/512)；low detail 恒 85。
/// - Anthropic：宽×高 / 750。
/// - DeepSeek / Ollama：不支持 → 0（不计）。
pub fn estimate_image_tokens(
    provider: ProviderKind,
    width: u32,
    height: u32,
    high_detail: bool,
) -> u64 {
    match provider {
        ProviderKind::OpenAi | ProviderKind::DeepSeek => {
            if !high_detail {
                return 85;
            }
            // 缩放：长边 ≤2048；再把短边缩到 ≤768（官方口径的最简实现）
            let mut w = f64::from(width);
            let mut h = f64::from(height);
            if w.max(h) > 2048.0 {
                let s = 2048.0 / w.max(h);
                w *= s;
                h *= s;
            }
            if w.min(h) > 768.0 {
                let s = 768.0 / w.min(h);
                w *= s;
                h *= s;
            }
            let tiles = (w / 512.0).ceil() as u64 * (h / 512.0).ceil() as u64;
            85 + 170 * tiles
        }
        // 官方口径四舍五入（文档示例 1092×1092 ≈ 1590 token）：(w*h + 375) / 750
        ProviderKind::Anthropic => (u64::from(width) * u64::from(height) + 375) / 750,
        ProviderKind::Ollama | ProviderKind::Custom => 0,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn bpe_init_failure_maps_to_inspector_error() {
        static BAD: LazyLock<Result<CoreBPE, String>> = LazyLock::new(|| Err("boom".into()));
        assert!(matches!(
            bpe(&BAD, "bad"),
            Err(InspectorError::InvalidInput(_))
        ));
    }

    #[test]
    fn estimate_text_is_positive_and_monotone() {
        let short = estimate_text("gpt-4o", "hello").unwrap();
        let long =
            estimate_text("gpt-4o", "hello world, this is a longer sentence indeed").unwrap();
        assert!(short >= 1);
        assert!(long > short);
    }

    #[test]
    fn estimate_text_empty_is_zero() {
        assert_eq!(estimate_text("deepseek-chat", "").unwrap(), 0);
    }

    #[test]
    fn openai_image_low_detail_flat() {
        assert_eq!(
            estimate_image_tokens(ProviderKind::OpenAi, 3000, 2000, false),
            85
        );
    }

    #[test]
    fn openai_image_high_detail_tile_formula() {
        // 1024x1024：无缩放，tiles = 2*2 = 4 → 85 + 170*4 = 765
        assert_eq!(
            estimate_image_tokens(ProviderKind::OpenAi, 1024, 1024, true),
            765
        );
        // 2048x4096：长边缩到 2048 → 1024x2048；短边 1024>768 缩 → 768x1536
        // tiles = ceil(768/512)*ceil(1536/512) = 2*3 = 6 → 85 + 1020 = 1105
        assert_eq!(
            estimate_image_tokens(ProviderKind::OpenAi, 2048, 4096, true),
            1105
        );
    }

    #[test]
    fn anthropic_image_area_formula() {
        // 1092x1092 → 1092*1092/750 = 1590.9 → 1590（整除截断）
        assert_eq!(
            estimate_image_tokens(ProviderKind::Anthropic, 1092, 1092, true),
            1590
        );
    }

    #[test]
    fn unsupported_providers_yield_zero() {
        assert_eq!(
            estimate_image_tokens(ProviderKind::Ollama, 512, 512, true),
            0
        );
    }
}
