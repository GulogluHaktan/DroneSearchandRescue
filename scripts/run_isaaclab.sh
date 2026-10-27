#!/usr/bin/env bash
# Run a script inside the Isaac Lab docker image (same recipe as DroneRacing / LunarSim).
# Usage: scripts/run_isaaclab.sh isaac/fidelity_test.py [args...]
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FORESTSAR_ISAACLAB_IMAGE:-lunarsim-isaaclab:6.0.1}"
CACHE_ROOT="${LUNARSIM_ISAAC_CACHE:-$HOME/docker/isaac-sim}"
mkdir -p "$CACHE_ROOT"/cache/{ov,pip,glcache,kit} "$CACHE_ROOT"/{data,documents,pylibs} "$PROJECT_ROOT/runs"
SCRIPT="$1"; shift
docker run --rm -e NOARROW="${NOARROW:-}" -e SEG_DEBUG="${SEG_DEBUG:-}" --gpus all --network=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONUNBUFFERED=1 \
  -v "$PROJECT_ROOT":/workspace/DroneTubitak1002 \
  -v "$CACHE_ROOT/cache/ov":/root/.cache/ov \
  -v "$CACHE_ROOT/cache/pip":/root/.cache/pip \
  -v "$CACHE_ROOT/cache/glcache":/root/.cache/nvidia \
  -v "$CACHE_ROOT/cache/kit":/root/.cache/kit \
  -v "$CACHE_ROOT/data":/root/.local/share/ov \
  -v "$CACHE_ROOT/documents":/root/Documents \
  -v "$CACHE_ROOT/pylibs":/root/.local \
  --entrypoint /workspace/IsaacLab/_isaac_sim/python.sh \
  -w /workspace/IsaacLab \
  "$IMAGE" "/workspace/DroneTubitak1002/$SCRIPT" --headless "$@"
chmod -R a+rwX "$PROJECT_ROOT/runs" 2>/dev/null || true
