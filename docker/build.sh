#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

PLATFORM="${1:-gb10}"

echo "============================================"
echo "  Building Cosmos3-Nano Docker: $PLATFORM"
echo "============================================"

case "$PLATFORM" in
    gb10)
        docker build \
            -f docker/Dockerfile.gb10 \
            -t cosmos3-nano:gb10 \
            .
        echo ""
        echo "Built: cosmos3-nano:gb10"
        echo "Run:   docker/run.sh gb10"
        ;;
    jetson-thor|thor)
        docker build \
            -f docker/Dockerfile.jetson-thor \
            -t cosmos3-nano:jetson-thor \
            .
        echo ""
        echo "Built: cosmos3-nano:jetson-thor"
        echo "Run:   docker/run.sh jetson-thor"
        ;;
    both)
        echo "Building GB10..."
        docker build -f docker/Dockerfile.gb10 -t cosmos3-nano:gb10 .
        echo ""
        echo "Building Jetson Thor..."
        docker build -f docker/Dockerfile.jetson-thor -t cosmos3-nano:jetson-thor .
        echo ""
        echo "Built both images."
        ;;
    *)
        echo "Usage: $0 [gb10|jetson-thor|both]"
        exit 1
        ;;
esac
