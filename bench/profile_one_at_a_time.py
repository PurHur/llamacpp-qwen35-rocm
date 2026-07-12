#!/usr/bin/env python3
"""PP/TPS benchmark for llama-server. Must be invoked via scripts/bench-pp-tps.sh (flock lock)."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URL = os.environ.get("QWEN35_BENCH_URL", "http://127.0.0.1:50126")
DEFAULT_MODEL = os.environ.get("QWEN35_BENCH_MODEL", "Qwythos-9B-v2-MTP-Q8_0.gguf")
BASE_PROMPT = (
    "You are a concise assistant.\n"
    "Explain what token throughput means for LLM inference in 2 short sentences.\n"
    "Do not use bullet points."
)


def wait_ready(url: str, timeout_s: int) -> None:
    health = f"{url.rstrip('/')}/health"
    deadline = time.time() + timeout_s
    last_err: str | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=10) as resp:
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2)
    raise RuntimeError(f"server not ready after {timeout_s}s (last: {last_err})")


def post_chat(url: str, model: str, messages: list[dict[str, str]], max_tokens: int, temperature: float, timeout_s: int) -> dict[str, Any]:
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
    out: dict[str, Any] = json.loads(body.decode("utf-8"))
    out["_wall_s"] = time.perf_counter() - t0
    return out


def extract_metrics(resp: dict[str, Any]) -> dict[str, Any]:
    usage = resp.get("usage") or {}
    timings = resp.get("timings") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "pp_tok_s": float(timings["prompt_per_second"]) if timings.get("prompt_per_second") is not None else None,
        "decode_tok_s": float(timings["predicted_per_second"]) if timings.get("predicted_per_second") is not None else None,
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
        "wall_s": float(resp.get("_wall_s") or 0),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Qwythos/qwen35 PP+TPS benchmark (requires profile lock)")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--title", default=os.environ.get("QWEN35_BENCH_TITLE", "qwen35-rocm"))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--json-out", default="", help="Write summary JSON here")
    args = p.parse_args()

    user_content = ((BASE_PROMPT.strip() + "\n\n") * args.repeat).strip()
    messages = [{"role": "user", "content": user_content}]
    url = args.url.rstrip("/")

    print(f"=== {args.title} ===")
    print(f"  URL: {url}  model: {args.model}  runs: {args.runs}  lock: held\n")
    wait_ready(url, 600)

    for i in range(args.warmup):
        post_chat(url, args.model, messages, args.max_tokens, args.temperature, 600)
        print(f"  warmup {i + 1}/{args.warmup}: ok")

    rows: list[dict[str, Any]] = []
    for i in range(args.runs):
        resp = post_chat(url, args.model, messages, args.max_tokens, args.temperature, 600)
        m = extract_metrics(resp)
        rows.append(m)
        pp = m["pp_tok_s"]
        tps = m["decode_tok_s"]
        print(
            f"  run {i + 1}: PP {pp:.1f} tok/s  decode {tps:.1f} tok/s  "
            f"wall {m['wall_s']:.2f}s  draft {m.get('draft_n')}/{m.get('draft_n_accepted')}"
        )

    pp_vals = [r["pp_tok_s"] for r in rows if r["pp_tok_s"] is not None]
    tps_vals = [r["decode_tok_s"] for r in rows if r["decode_tok_s"] is not None]
    summary = {
        "title": args.title,
        "url": url,
        "model": args.model,
        "pp_mean": statistics.mean(pp_vals) if pp_vals else None,
        "pp_stdev": statistics.pstdev(pp_vals) if len(pp_vals) > 1 else 0,
        "decode_mean": statistics.mean(tps_vals) if tps_vals else None,
        "decode_stdev": statistics.pstdev(tps_vals) if len(tps_vals) > 1 else 0,
        "runs": rows,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("\n=== Summary ===")
    print(f"  PP mean: {summary['pp_mean']:.1f} tok/s" if summary["pp_mean"] else "  PP: n/a")
    print(f"  Decode mean: {summary['decode_mean']:.1f} tok/s" if summary["decode_mean"] else "  Decode: n/a")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
