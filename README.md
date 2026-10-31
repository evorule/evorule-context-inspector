# evorule-context-inspector

> LLM context snapshot and health audit chain — structure what the agent saw at decision time into a standalone BLAKE3 audit chain, making "what the agent saw" verifiable, replayable, and private by default.

[![CI][badge-ci]][ci-link]
[![License: AGPL-3.0-or-later][badge-license]][license-link]

**License**: [AGPL-3.0-or-later](LICENSE) with [dual licensing](DUAL_LICENSE.md)

The engine's audit chain records rule-execution facts (which rule fired, what trace hit, the causal chain), but what the agent's context window looked like at decision time stays invisible. This repo adds that chain:

- **Request side** — `ContextSnapshot`: the context frozen just before an LLM call goes out (token counts + structural skeleton)
- **Response side** — `ResponseObservation`: appended after the reply (real usage / finish reason / truncation detection), paired via `response_to`
- **Five health dimensions** — pressure (four left-closed bands), cache hit rate, drift (multiset Jaccard plus explicit SystemPromptChanged / ToolSetChanged / ContextReset detections), truncation, and repetition (word-shingle); all stored as basis-point integers
- **Standalone BLAKE3 audit chain** — chain steps reuse the reactor's `chain_step`; the chain head is periodically anchored into the governance Ed25519 anchor stream
- **Plugin form** — a manifest-type plugin running as a standalone process (MITM listener + audit chain + context API + `/admin/bindings`), with zero mechanism changes in evorule-server

## Privacy model (strong privacy by default)

| Layer | Content | Default storage |
|---|---|---|
| Per-item content hash | BLAKE3 of each system / tool / message original | hash only, never the original text |
| `skeleton_hash` | canonical JSON structural skeleton | always stored — recomputable from storage alone, the strong semantics of chain verification |
| `content_hash` | hash of the full messages + tools originals | only with `--store-content`, kept in a sidecar file, **never chained** |

No hash input ever includes HTTP headers (`Authorization` / `x-api-key` never enter hashes, logs, or disk).

Hashing uses [evorule-hash](https://github.com/evorule/evorule-hash) primitives (`digest` / `json_digest`) — no self-rolled blake3; canonical JSON is serde_json's default BTreeMap key order (`preserve_order` off). This repo's hash domain is registered in the hash crate's domain registry.

> [!NOTE]
> **Status: Phase 0 (skeleton and registry registration) is complete.** Later phases — data model, providers and streaming, chain and plugin integration, UI, docs — land incrementally; the in-repo code is the single source of truth, and implementation docs land in `docs/` as phases progress.

## Prerequisites

- Rust toolchain (pinned by `rust-toolchain.toml`)
- Sibling checkouts of `evorule` and `evorule-hash` — the kernel crates are path dependencies during the skeleton phase

## Build

```bash
cargo build
cargo test
```

## Run

The plugin process (MITM listener + context API + `/admin/bindings`) listens on the 19100 port range, loopback-only — it never exposes a network interface.

## License

**AGPL-3.0-or-later**: [LICENSE](LICENSE) / [NOTICE.md](NOTICE.md); dual-licensing guidance in [DUAL_LICENSE.md](DUAL_LICENSE.md), with [FREE_COMMERCIAL_LICENSE.md](FREE_COMMERCIAL_LICENSE.md) (free waiver for eligible entities) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

[badge-ci]: https://github.com/evorule/evorule-context-inspector/actions/workflows/ci.yml/badge.svg
[ci-link]: https://github.com/evorule/evorule-context-inspector/actions/workflows/ci.yml
[badge-license]: https://img.shields.io/badge/license-AGPL--3.0--or--later-blue?style=flat-square
[license-link]: LICENSE
