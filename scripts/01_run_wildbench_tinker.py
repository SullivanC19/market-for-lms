# run_wildbench_tinker.py
#
# Run all WildBench questions for 10 stochastic repetitions across the first
# 7 cheaper representative Tinker generators.
#
# Each model/repetition writes one WildBench-compatible local-results JSON:
#   results/wild_bench_v2/passk10/<generator>/rep_00.json
#   ...

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import constants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-script", default=constants.DEFAULT_GENERATOR_DRIVER_SCRIPT)
    parser.add_argument("--generators-file", default=constants.DEFAULT_GENERATORS_REGISTRY)
    parser.add_argument("--responses-dir", default=constants.DEFAULT_RESPONSES_DIR)
    parser.add_argument("--repetitions", type=int, default=constants.DEFAULT_REPETITIONS)
    parser.add_argument("--temperature", type=float, default=constants.DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=constants.DEFAULT_MAX_TOKENS)
    parser.add_argument("--concurrency", type=int, default=constants.DEFAULT_CONCURRENCY)
    parser.add_argument("--request-timeout-s", type=float, default=constants.DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--retries", type=int, default=constants.DEFAULT_RETRIES_TINKER_RUNNER)
    parser.add_argument("--limit", type=int, default=None, help="Debug only. Omit for all 1024 WildBench tasks.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    generator_script = Path(args.generator_script)
    if not generator_script.exists():
        raise FileNotFoundError(f"Missing generator script: {generator_script}")

    for generator in constants.DEFAULT_GENERATORS:
        cmd = [
            sys.executable,
            str(generator_script),
            "--generator", generator,
            "--generators-file", args.generators_file,
            "--responses-dir", args.responses_dir,
            "--repetitions", str(args.repetitions),
            "--temperature", str(args.temperature),
            "--max-tokens", str(args.max_tokens),
            "--concurrency", str(args.concurrency),
            "--request-timeout-s", str(args.request_timeout_s),
            "--retries", str(args.retries),
        ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.verbose:
            cmd.append("--verbose")

        print("\n" + "=" * 100)
        print(f"Running {generator}")
        print(" ".join(cmd))
        print("=" * 100)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
