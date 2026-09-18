#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# SSH tunnel helper — connects your laptop to the GPU server on the cluster
#
# Usage:
#   ./scripts/tunnel.sh <compute-node-hostname>
#
# Get the compute node from the SLURM job output:
#   tail -f logs/gpu_server_<JOBID>.out
#
# Example:
#   ./scripts/tunnel.sh ggpu021
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NODE="${1:?Usage: $0 <compute-node-hostname>}"
GWDG_USER="u29661"
LOGIN_NODE="glogin-gpu.hpc.gwdg.de"
LOCAL_PORT=8000
REMOTE_PORT=8000

echo "Opening tunnel: localhost:${LOCAL_PORT} → ${NODE}:${REMOTE_PORT}"
echo "Press Ctrl+C to close."
echo ""
echo "While the tunnel is open, set in config/settings.yaml:"
echo "  gpu:"
echo "    remote_url: \"http://localhost:${LOCAL_PORT}\""
echo ""
echo "When done, close the tunnel and reset remote_url to \"\" in settings.yaml."

# Without keepalives, a network drop leaves a zombie tunnel that accepts
# local connections but never forwards them (silent timeouts on the client)
ssh -N \
    -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
    -J "${GWDG_USER}@${LOGIN_NODE}" \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    "${GWDG_USER}@${NODE}"
