# proxycache (llama-swap fork)

Fork of [airnsk/proxycache](https://github.com/airnsk/proxycache) with added support for llama-swap and automatic cache cleanup.

## Changes from upstream

**llama-swap compatibility:**
- Routes `/slots` API calls through `/upstream/{model}/slots/` path
- Reads model ID from request body instead of `/v1/models` endpoint
- Passes through `/v1/models` from backend for proper model discovery

**Cache cleanup:**
- Automatic periodic cleanup based on age and/or total size
- Configurable via environment variables
- Runs in background, no cronjob needed

## Requirements

- Python 3.10+
- llama.cpp server with `--slot-save-path` configured
- llama-swap (optional, but recommended for multi-model setups)

## Installation

```bash
git clone https://github.com/dingausmwald/proxycache.git
cd proxycache
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

All configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_URL` | `http://127.0.0.1:8000` | Backend URL (llama-swap or llama-server) |
| `N_SLOTS` | `2` | Number of slots per backend |
| `PORT` | `8081` | Port proxycache listens on |
| `META_DIR` | `./kv_meta` | Directory for .meta.json files |
| `BIG_THRESHOLD_WORDS` | `500` | Min words to trigger caching |
| `WORDS_PER_BLOCK` | `100` | Words per block for LCP matching |
| `LCP_TH` | `0.6` | LCP similarity threshold (0-1) |
| `REQUEST_TIMEOUT` | `600` | HTTP timeout in seconds |
| `CACHE_DIR` | `` | llama.cpp cache dir (for cleanup) |
| `CACHE_MAX_AGE_HOURS` | `168` | Delete files older than this (0=disabled) |
| `CACHE_MAX_SIZE_GB` | `50` | Max total cache size |
| `CACHE_CLEANUP_INTERVAL_MINUTES` | `60` | Cleanup check interval |

## Quick start

```bash
# Start with llama-swap backend
LLAMA_URL="http://127.0.0.1:9292" \
N_SLOTS=1 \
PORT=5000 \
META_DIR="/path/to/proxycache-meta" \
CACHE_DIR="/path/to/kv-cache" \
BIG_THRESHOLD_WORDS=1500 \
python3 proxycache.py
```

## systemd service

Create `~/.config/systemd/user/proxycache.service`:

```ini
[Unit]
Description=ProxyCache for llama.cpp KV Cache Management
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/proxycache
Environment="LLAMA_URL=http://127.0.0.1:9292"
Environment="N_SLOTS=1"
Environment="META_DIR=/path/to/proxycache-meta"
Environment="PORT=5000"
Environment="BIG_THRESHOLD_WORDS=1500"
Environment="CACHE_DIR=/path/to/kv-cache"
Environment="CACHE_MAX_AGE_HOURS=0"
Environment="CACHE_MAX_SIZE_GB=100"
ExecStart=/path/to/proxycache/venv/bin/python proxycache.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now proxycache
```

## Usage with llama-swap

Architecture:

```
Client (OpenWebUI, Kilo Code, etc.)
    |
    v
proxycache (:5000) - KV cache management
    |
    v
llama-swap (:9292) - model routing
    |
    v
llama-server (:PORT) - inference
```

Make sure your llama-swap model configs include `--slot-save-path`:

```yaml
models:
  "my-model":
    cmd: "llama-server -m model.gguf --slot-save-path /path/to/kv-cache ..."
```

Point your clients to proxycache (port 5000) instead of llama-swap directly.

## How it works

1. Incoming request hits proxycache
2. Prompt is hashed and compared against cached prefixes (LCP matching)
3. If match found: restore KV cache from disk, skip prefill
4. If no match: normal inference, then save KV cache to disk
5. Periodic cleanup removes old/excess cache files

Cache files are stored by llama-server in `CACHE_DIR`, meta files (for LCP lookup) in `META_DIR`.

## License

Same as upstream - see original repo.
