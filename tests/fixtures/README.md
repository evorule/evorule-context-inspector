# fixtures 来源与口径

验收矩阵 #1-#4 的录制物料。**来源分两档**，每档的证明力不同：

| 文件 | 来源 | 说明 |
|------|------|------|
| `openai_request.json` | **真实发出** | 2026-10-02 经 MiniMax OpenAI 兼容端点实发（与 `openai_nonstream.json` 同请求） |
| `openai_nonstream.json` | **真实录制** | MiniMax-M2.5（OpenAI 协议兼容）非流式响应原文，含真实 usage |
| `openai_stream.sse` | **真实录制** | 同端点流式响应原文；`stream_options.include_usage` 已注入，末 chunk 带真实 usage |
| `anthropic_request.json` / `_nonstream.json` / `_stream.sse` | **spec 派生** | 本机无 Anthropic 凭据；按官方 Messages API 文档形态构造（含 cache_read_input_tokens / message_delta 双字段 chunk）。取得凭据后以真实录制替换（登记待办） |
| `deepseek_request.json` / `_nonstream.json` / `_stream.sse` | **spec 派生** | 本机无 DeepSeek 凭据；协议形态与 cache 字段（`prompt_cache_hit_tokens`）按官方文档构造 |
| `ollama_request.json` / `_nonstream.json` / `_stream.ndjson` | **spec 派生** | 录制时本机 Ollama 未运行；/api/chat NDJSON 形态按官方文档构造 |

约束：

- fixture 文件**永不包含** API 密钥或敏感 header（§2.5 密钥安全承诺；密扫覆盖本目录）。
- 真实录制件不得手工修改（重录替换，保持证据链）。
- spec 派生件升级为真实件时，同步更新本表与验收测试中的来源断言。
