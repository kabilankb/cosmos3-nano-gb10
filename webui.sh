#!/bin/sh
# Launch Cosmos3-Nano Web UI in Docker
# Usage: ./webui.sh [port]  (default: 7860)
exec "$(dirname "$0")/docker/run.sh" gb10 webui "${1:-7860}"
