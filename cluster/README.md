# GPU Cluster Guide (GWDG Grete)

Everything needed to run the BGE models on GWDG Grete.
All commands run from your **laptop terminal**, in the **repo root**.

---

## The Big Picture (read this once)

Three machines are involved:

| Where | What it has | Internet |
|---|---|---|
| **Your laptop** | code, Qdrant (Docker), results | yes |
| **Login node** (`glogin-gpu.hpc.gwdg.de`) | gateway you SSH into; `~` is SHARED with compute nodes | yes |
| **Compute nodes** (`ggpuXXX`) | the GPUs | **NO** |

Key facts:

- **SLURM is the booking system.** You submit a *job* (a script + a time
  limit). When a GPU node is free, SLURM runs your job there and **holds the
  GPU for the whole job** — you do not need to watch or refresh anything.
- **Your reservation = a running job.** It lasts until the job finishes, you
  cancel it, or its time limit hits.
- **Two jobs exist for this project:**
  - `cluster/slurm_batch_embed.sh` — one-shot: embeds all chunks, writes
    `vectors.json`. ~2 min on the GPU.
  - `cluster/slurm_interactive.sh` — session: runs the GPU server for **6 h**
    (set via `--time` in the script). For live queries/validation.
- **Models are pre-cached** in `~/.cache/huggingface` on the cluster (6.5 GB,
  shared home). Scripts set `HF_HUB_OFFLINE=1` so jobs never touch the
  internet. (If you ever download a NEW model, do it on the **login node** —
  compute nodes cannot reach huggingface.co.)
- **Partitions used:** `kisski` (A100, for batch jobs) and
  `grete:interactive` (shared A100 slice, for sessions). Do not change these
  without checking access with `sinfo`.

---

## The Daily Loop (validation session, start to finish)

The full sequence when you push changes and want fresh validation numbers.
Every command runs on the **laptop** unless the comment says `cluster`.

```bash
# 1. laptop: commit + push
git add -A && git commit -m "..." && git push

# 2. cluster: pull (if you changed code) + submit the 6 h session job
ssh u29661@glogin-gpu.hpc.gwdg.de 'cd ~/thesis-rag && git pull && sbatch cluster/slurm_interactive.sh'
# → "Submitted batch job <ID>" — remember the ID
# (If you pushed data/config only, the pull is harmless anyway — do it, it's 3 s)

# 3. check status — as often as you like
./scripts/check_gpu.sh <ID>
# PENDING → shows planned node + estimated start time. Nothing to do; it auto-starts.
# RUNNING → shows node + log tail.

# 4. wait for readiness: the log must say "Both models ready."
#    (models take ~1–2 min to load after the job starts)
./scripts/check_gpu.sh <ID>

# 5. laptop: open the tunnel (keep this terminal open the whole session)
./scripts/tunnel.sh <node>

# 6. laptop, other terminal: verify the tunnel actually works
curl -s http://localhost:8000/health     # must print {"status":"ok"}
# (give the tunnel ~10 s to finish connecting after launch — it prints its
#  banner before ssh has bound the port)

# 7. laptop: flip the switch, then run
#    config/settings.yaml → gpu.remote_url: "http://localhost:8000"
#    DO THIS ONLY AFTER step 6 printed {"status":"ok"}. It tells the pipeline
#    to send model calls through your tunnel instead of looking for local GPUs.
#    (It's "" the rest of the time — see teardown below.)
.venv/bin/python scripts/validate_retrieval.py

# 8. tear down (in this order)
#    a) Ctrl+C in the tunnel terminal
#    b) ssh u29661@glogin-gpu.hpc.gwdg.de 'scancel <ID>'
#    c) config/settings.yaml → gpu.remote_url: ""
#       (forget this and the next laptop run will fail with "Connection
#        refused" on localhost:8000)
```

---

## The Commands

### A. One-shot embedding (after re-parsing/re-chunking documents)

```bash
# 1. make fresh chunks on the laptop (no GPU needed)
.venv/bin/python scripts/ingest.py --export-chunks data/chunks_export.json

# 2. upload chunks to the cluster
scp data/chunks_export.json u29661@glogin-gpu.hpc.gwdg.de:~/thesis-rag/data/chunks_export.json

# 3. submit the GPU job
ssh u29661@glogin-gpu.hpc.gwdg.de 'cd ~/thesis-rag && sbatch cluster/slurm_batch_embed.sh data/chunks_export.json data/vectors.json'
```
→ prints `Submitted batch job <ID>`. **Remember the ID.**

### B. Reserve a GPU for live queries (6 h session)

```bash
ssh u29661@glogin-gpu.hpc.gwdg.de 'cd ~/thesis-rag && sbatch cluster/slurm_interactive.sh'
```
→ prints `Submitted batch job <ID>`. **Remember the ID.**

### C. Check any job's status (the command you'll use most)

```bash
./scripts/check_gpu.sh <ID>        # one-liner: state + node + log tail
ssh u29661@glogin-gpu.hpc.gwdg.de 'squeue --me --start'   # raw version
```

| What you see | Meaning |
|---|---|
| `PENDING` / `PD` | waiting for a free node — **nothing to do**, SLURM starts it automatically. With `--start`, the `START_TIME` column is SLURM's **estimated** start (can drift if the node stays busy); the last column is the wait reason, e.g. `(TopOfQueue)` = you're first in line |
| `RUNNING` + node name (e.g. `ggpu021`) | **the GPU is yours**; the time column = how long it has run |
| *(no rows)* | job is over (finished, failed, or cancelled) — check the log (D) |

**"GPU is yours" is not "ready to use".** Two more steps:
1. the log must contain `Both models ready.` (model loading takes ~1–2 min)
2. your tunnel must be up and `curl -s http://localhost:8000/health` must return `{"status":"ok"}`

**Time left on a reservation** = `6:00:00` − TIME (for interactive jobs).
Need more? Submit command B again later — a fresh 6 h.

**How busy is the partition?** (why a job might wait)
```bash
ssh u29661@glogin-gpu.hpc.gwdg.de 'squeue -p kisski --noheader | wc -l; sinfo -p kisski'
```

### D. See what a job is doing / did

```bash
# main output (progress lines)
ssh u29661@glogin-gpu.hpc.gwdg.de 'tail -20 ~/thesis-rag/logs/<jobname>_<ID>.out'
# errors / model loading / tracebacks
ssh u29661@glogin-gpu.hpc.gwdg.de 'tail -20 ~/thesis-rag/logs/<jobname>_<ID>.err'
```
Job names: `batch_embed` (batch job), `gpu_server` (interactive job).

Success for a batch job = last line of `.out` says
`Wrote 749 vectors → data/vectors.json` (number = your chunk count).

### E. Get notified instead of watching (run in a laptop terminal)

**One-shot batch job** — notify when it *finishes* (replace `<ID>`), then leave
the terminal alone:

```bash
JOBID=<ID>; while ssh -q u29661@glogin-gpu.hpc.gwdg.de "squeue -j $JOBID --noheader" | grep -q .; do sleep 60; done; osascript -e "display notification \"GPU job $JOBID is done — check its log\" with title \"SLURM\""
```

**6 h session** — you need to know when it *starts* (that's when you can
tunnel in). Notifies when it becomes `RUNNING`, or if the job vanishes:

```bash
JOBID=<ID>; while true; do s=$(ssh -q u29661@glogin-gpu.hpc.gwdg.de "squeue -j $JOBID --noheader"); [ -z "$s" ] && { osascript -e "display notification \"job vanished (cancelled/failed)\" with title \"SLURM\""; break; }; echo "$s" | grep -q RUNNING && { osascript -e "display notification \"GPU session RUNNING — get node name from squeue\" with title \"SLURM\""; break; }; sleep 30; done
```

### F. Pull results from the cluster

```bash
# batch job output → laptop
scp u29661@glogin-gpu.hpc.gwdg.de:~/thesis-rag/data/vectors.json data/vectors.json

# then store in Qdrant (Qdrant must be running: docker start thesis-qdrant)
.venv/bin/python scripts/ingest.py --vectors-file data/vectors.json
```

### G. Use the reserved GPU for live queries

Do these **in order**, each in its own terminal:

1. Terminal 1 — connect (node name from command C). **Keep open.**
   ```bash
   ./scripts/tunnel.sh ggpu021
   ```
2. Terminal 2 — verify the tunnel works, THEN set the switch:
   ```bash
   curl -s http://localhost:8000/health   # wait for {"status":"ok"}
   ```
   Only after that, in `config/settings.yaml`:
   `gpu.remote_url: "http://localhost:8000"`, then:
   ```bash
   .venv/bin/python scripts/validate_retrieval.py
   ```
   First query takes ~30–60 s (models warming up); after that each query is
   a couple of seconds. Remember to reset `remote_url: ""` at teardown (H).

   Note: one full run makes 95 SAIA rewrite calls (the LLM is rate-limited).
   If you hit limit errors, add a small sleep between queries in
   `scripts/validate_retrieval.py` and/or retry-with-backoff on 429s.

### H. End / release

```bash
# 1. Ctrl+C in the tunnel terminal
# 2. release the GPU
ssh u29661@glogin-gpu.hpc.gwdg.de 'scancel <ID>'
# 3. reset config
#    config/settings.yaml → gpu.remote_url: ""
```

---

## When things go wrong

| Symptom | What it is / do |
|---|---|
| Job sits `PENDING` a long time | Partition busy. No reliable ETA exists. Use the notifier (E). Check who's queued: `ssh ... 'squeue -p kisski \| head'` |
| Job finished but no result file | Read the `.err` log (D) — last lines contain the traceback |
| `Invalid account or account/partition combination` | Your account can't use that partition. You (KISSKI) can use `kisski`, `kisski-h100`, `grete:interactive` — not `grete:shared` |
| `[Errno 101] Network is unreachable` + huggingface retries | Compute node tried the internet. Make sure the script exports `HF_HUB_OFFLINE=1`. If a NEW model is missing, download it on the **login node**: `ssh ... '~/thesis-venv-gpu/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(\"<org>/<model>\")"'` |
| Tunnel says connection refused | Node name wrong, server not up yet, or tunnel still connecting. Wait ~10 s after launching the tunnel, then `curl -s http://localhost:8000/health`. Check C for `RUNNING` + "Both models ready." in log D |
| Tunnel zombie: port 8000 accepts connections but requests time out | Network blip killed the path to the cluster. Kill it (`lsof -ti tcp:8000 \| xargs kill`) and re-run `./scripts/tunnel.sh <node>`. Keepalives in tunnel.sh now make the tunnel die on its own after ~45 s of silence — a hanging tunnel means you closed the wrong terminal or the node went down |
| Laptop scripts fail with `Connection refused` / `httpx` errors | `remote_url` is set but no tunnel/GPU session is active. Start one (B/G) or reset `remote_url: ""` |
| `ValueError ... require users to upgrade torch to at least v2.6` | venv torch too old to load the cached `.bin` models. On the **login node**: `~/thesis-venv-gpu/bin/pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124` (cu124, not cu121 — cu121 wheels stop at torch 2.5.1; Grete drivers 570 support cu124) |
| `Qdrant not reachable` / `ConnectionError` on 6333 | `docker start thesis-qdrant`. If the container is gone entirely: `docker run -d -p 6333:6333 --name thesis-qdrant -v thesis-qdrant-data:/qdrant/storage qdrant/qdrant` |
| Session died mid-work (squeue empty) | Node issue or 6 h time limit hit. Check logs D, then re-submit B (fresh 6 h) and redo G |
| You changed code locally | `git push`, then `ssh ... 'cd ~/thesis-rag && git pull'` — cluster copy is a git clone |
| Job killed with `TIME LIMIT REACHED` | It ran the full `--time`. Increase `--time` in the script (max 12 h on these partitions) |

---

## Where everything lives

| Thing | Location |
|---|---|
| PDFs, `chunks_export.json`, `vectors.json`, results | laptop `data/` |
| Qdrant index | laptop, Docker container `thesis-qdrant` |
| Code | both, via git (active month branch, currently `month-3/system-a-single-pass-rag`) |
| GPU models (cache) | cluster `~/.cache/huggingface` (shared home) |
| GPU Python env (torch 2.6+cu124) | cluster `~/thesis-venv-gpu` |
| Job logs | cluster `~/thesis-rag/logs/` |
