//! canonical JSON 封装 —— 链哈希输入的确定性纪律执行点（设计稿 §2.5/§2.7）。
//!
//! - **键序**：serde_json 默认 `Map = BTreeMap`，序列化输出键字典序。
//!   依赖树任何一方开启 `preserve_order` feature（Map 变 IndexMap，保留插入序）
//!   都会被本模块的键序守护测试当场打红 —— 这是 R3 的守护点，不靠自觉。
//! - **整数纪律**：链哈希输入只允许整数与字符串；任何浮点（含整数值的
//!   `2.0` 字面量）一律拒收 —— 浮点的最短往返表示虽在实践中稳定，
//!   但纪律上不入链，`FloatForbidden` fail-fast。
//! - **紧凑**：输出为紧凑 JSON（无空白），与 reactor `fact_to_stable_json`
//!   的序列化形态同构（`serde_json::to_string`）。

use crate::InspectorError;
use serde::Serialize;
use serde_json::Value;

/// 把任意可序列化值转为 canonical `serde_json::Value`（键已按字典序排列），
/// 浮点一律拒收。
pub fn to_canonical_value<T: Serialize>(value: &T) -> Result<Value, InspectorError> {
    let v = serde_json::to_value(value)
        .map_err(|e| InspectorError::InvalidInput(format!("序列化失败：{e}")))?;
    reject_floats(&v, "$")?;
    Ok(v)
}

/// canonical JSON 字符串（紧凑、键字典序、无浮点）。
pub fn canonical_json<T: Serialize>(value: &T) -> Result<String, InspectorError> {
    let v = to_canonical_value(value)?;
    serde_json::to_string(&v).map_err(|e| InspectorError::InvalidInput(format!("序列化失败：{e}")))
}

fn reject_floats(value: &Value, path: &str) -> Result<(), InspectorError> {
    match value {
        Value::Number(n) => {
            if n.is_f64() {
                Err(InspectorError::FloatForbidden {
                    path: path.to_owned(),
                })
            } else {
                Ok(())
            }
        }
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                reject_floats(item, &format!("{path}[{i}]"))?;
            }
            Ok(())
        }
        Value::Object(map) => {
            for (k, v) in map {
                reject_floats(v, &format!("{path}.{k}"))?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use serde_json::json;

    #[test]
    fn keys_are_lexicographically_sorted() {
        // 键插入顺序刻意乱序：守护 preserve_order 被开启的情况
        // （serde_json Map 退化为 IndexMap 时本测试立即变红）。
        let v = json!({"z": 1, "a": 2, "m": {"y": 1, "b": 2}});
        let s = canonical_json(&v).unwrap();
        assert_eq!(s, r#"{"a":2,"m":{"b":2,"y":1},"z":1}"#);
    }

    #[test]
    fn floats_are_rejected_with_path() {
        let err = canonical_json(&json!({"a": {"b": 1.5}})).unwrap_err();
        assert!(
            matches!(err, InspectorError::FloatForbidden { ref path } if path == "$.a.b"),
            "期望 FloatForbidden($.a.b)，实际 {err:?}"
        );
    }

    #[test]
    fn null_bool_and_nested_containers_pass() {
        assert_eq!(canonical_json(&json!(null)).unwrap(), "null");
        assert_eq!(canonical_json(&json!(true)).unwrap(), "true");
        let v = json!({"a": [true, null, {"b": [1, null]}]});
        assert_eq!(
            canonical_json(&v).unwrap(),
            r#"{"a":[true,null,{"b":[1,null]}]}"#
        );
    }

    /// 序列化自身即失败的输入 → InvalidInput（不 panic、不静默）。
    struct AlwaysErr;
    impl serde::Serialize for AlwaysErr {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("boom"))
        }
    }

    #[test]
    fn serialization_failure_maps_to_invalid_input() {
        assert!(matches!(
            canonical_json(&AlwaysErr).unwrap_err(),
            InspectorError::InvalidInput(_)
        ));
    }

    #[test]
    fn integral_float_literal_is_rejected_too() {
        // 2.0 在 serde_json 中是 f64 —— 数值上整也不准入链（整数纪律）。
        assert!(canonical_json(&json!(2.0)).is_err());
        assert!(canonical_json(&json!({"t": [1, 2.0]})).is_err());
    }

    #[test]
    fn integers_and_strings_pass() {
        assert_eq!(canonical_json(&json!(2)).unwrap(), "2");
        assert_eq!(canonical_json(&json!(-7)).unwrap(), "-7");
        assert_eq!(canonical_json(&json!("文本")).unwrap(), "\"文本\"");
        assert_eq!(
            canonical_json(&json!({"n": -7, "s": "x"})).unwrap(),
            r#"{"n":-7,"s":"x"}"#
        );
    }

    #[test]
    fn serialization_is_deterministic_across_runs() {
        let v = json!({"model": "deepseek-chat", "provider": "deepseek", "n": 3});
        let a = canonical_json(&v).unwrap();
        let b = canonical_json(&v).unwrap();
        assert_eq!(a, b);
    }
}
