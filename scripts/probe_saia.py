"""Quick SAIA health probe: one trivial chat call, no retries.

Usage (from repo root):
    .venv/bin/python -u scripts/probe_saia.py

Prints "OK <s>s -> ..." and exits 0 if SAIA responds;
"FAIL <s>s -> <error>" and exits 1 if it does not.
A healthy SAIA answers the 10-token probe in well under 10 s.
"""
import time
from pathlib import Path

from openai import OpenAI


def _load_key() -> str | None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("SAIA_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> int:
    key = _load_key()
    if not key:
        print("FAIL: no SAIA_API_KEY in .env")
        return 1
    t0 = time.time()
    try:
        client = OpenAI(
            api_key=key,
            base_url="https://chat-ai.academiccloud.de/v1",
            timeout=60,
            max_retries=0,
        )
        r = client.chat.completions.create(
            model="qwen3.8-27b",
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=10,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        print(f"OK {time.time() - t0:.1f}s -> {r.choices[0].message.content!r}")
        return 0
    except Exception as e:
        print(f"FAIL {time.time() - t0:.1f}s -> {type(e).__name__}: {str(e)[:160]}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
