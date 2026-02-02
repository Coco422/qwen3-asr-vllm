#!/usr/bin/env python3
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _http_ready(url: str, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status:
                    return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            time.sleep(1)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the official Qwen3-ASR demo launcher (qwen-asr-demo).")
    parser.add_argument(
        "--asr-checkpoint",
        default="./model/Qwen3-ASR-1.7B",
        help="Local path or HF model id (default: ./model/Qwen3-ASR-1.7B)",
    )
    parser.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=300, help="Seconds to wait for the demo to become reachable.")
    parser.add_argument(
        "--stay",
        action="store_true",
        help="Keep the demo process running after it becomes reachable (no auto-shutdown).",
    )
    args = parser.parse_args()

    demo_bin = shutil.which("qwen-asr-demo")
    if not demo_bin:
        print("ERROR: qwen-asr-demo not found in PATH. Install the official package first (see README).", file=sys.stderr)
        return 1

    if (args.asr_checkpoint.startswith("./") or args.asr_checkpoint.startswith("/")) and not os.path.exists(
        args.asr_checkpoint
    ):
        print(f"ERROR: --asr-checkpoint not found: {args.asr_checkpoint}", file=sys.stderr)
        return 1

    cmd = [
        demo_bin,
        "--asr-checkpoint",
        args.asr_checkpoint,
        "--backend",
        args.backend,
        "--cuda-visible-devices",
        str(args.cuda_visible_devices),
        "--ip",
        args.ip,
        "--port",
        str(args.port),
    ]
    print("Launching:", " ".join(cmd), flush=True)

    proc = subprocess.Popen(cmd)

    def _shutdown() -> None:
        if proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
        except Exception:
            proc.kill()

    try:
        check_host = "127.0.0.1" if args.ip in ("0.0.0.0", "::") else args.ip
        url = f"http://{check_host}:{args.port}/"
        if not _http_ready(url, args.timeout):
            print(f"ERROR: demo not reachable within {args.timeout}s: {url}", file=sys.stderr)
            _shutdown()
            return 2

        print(f"OK: demo is reachable: {url}")
        if args.stay:
            print("Staying alive. Press Ctrl-C to stop.")
            proc.wait()
            return proc.returncode or 0
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if not args.stay:
            _shutdown()


if __name__ == "__main__":
    raise SystemExit(main())


