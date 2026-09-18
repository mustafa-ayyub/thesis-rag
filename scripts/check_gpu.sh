#!/usr/bin/env bash
# One-command GPU job status check.
#
# Usage: ./scripts/check_gpu.sh <jobid>
#
# Tells you: is the job waiting / running / over, which node it's on
# (or planned for), and whether the server has finished loading the models.
set -euo pipefail

JOBID="${1:?Usage: $0 <jobid>}"
GWDG_USER="u29661"
LOGIN_NODE="glogin-gpu.hpc.gwdg.de"

ssh "${GWDG_USER}@${LOGIN_NODE}" bash -s <<EOF
line=\$(squeue -j ${JOBID} --noheader)
if [ -z "\$line" ]; then
  echo "JOB OVER (finished / failed / cancelled). Last log lines:"
  tail -5 ~/thesis-rag/logs/gpu_server_${JOBID}.out 2>/dev/null || true
  tail -5 ~/thesis-rag/logs/gpu_server_${JOBID}.err 2>/dev/null || true
  exit 0
fi
echo "\$line"
state=\$(echo "\$line" | grep -oE 'PD|RUNNING|COMPLET' | head -1)
node=\$(echo "\$line" | grep -oE 'ggpu[0-9]+' | head -1)
if [ "\$state" = "PD" ]; then
  eta=\$(squeue -j ${JOBID} --noheader --start | grep -oE '20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
  echo "-> Waiting for a GPU node (\${node:-unknown} planned). Estimated start: \${eta:-unknown}. Nothing to do; SLURM starts it automatically."
else
  echo "-> Node: \$node. Server log tail (ready when it says 'Both models ready.'):"
  tail -4 ~/thesis-rag/logs/gpu_server_${JOBID}.out 2>/dev/null || echo "  (no log yet)"
fi
EOF
