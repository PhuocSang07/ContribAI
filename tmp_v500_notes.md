## What's New in v5.0.0

ContribAI fully rewritten in Rust with 100% feature parity to Python v4.1.0.

### ⚡ Performance vs Python v4

| Metric | Python v4 | Rust v5 |
|---|---|---|
| Startup time | ~800ms | ~5ms |
| Analysis speed | 1× | ~10–50× |
| Binary size | N/A | ~4.5 MB stripped |
| Memory usage | ~120 MB | ~8 MB |

### 🦀 21 CLI Commands Ported

`run` · `hunt` · `patrol` · `target` · `analyze` · `solve` · `stats` · `status` · `leaderboard` · `models` · `templates` · `profile` · `cleanup` · `notify-test` · `system-status` · `web-server` · `schedule` · `mcp-server` · `init` · `login` · `config-get/set/list`

### 🔧 Architecture

- **21 MCP tools** for Claude Desktop (stdio JSON-RPC)
- **GitHub GraphQL** + REST v3
- **API key auth** + HMAC-SHA256 webhook signatures
- **17 progressive analysis skills** (SQL injection, XSS, resource leak, etc.)
- **Docker sandbox** + local fallback
- **Event bus** — 15 typed events + JSONL logging
- **Working memory** — 72h TTL per repo (SQLite)

### 🧪 Tests

323 unit tests passing (mockall, wiremock, tokio-test)

### 📦 Install

```bash
cargo install --path crates/contribai-rs
contribai --version  # 5.0.0
```
