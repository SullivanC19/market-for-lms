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


GENERATORS = [
    "tinker__llama_3_2_1b",
    "tinker__qwen3_4b_instruct_2507",
    "tinker__qwen3_30b_a3b_instruct_2507",
    "tinker__gpt_oss_20b",
    "tinker__nemotron_3_nano_30b_a3b_bf16",
    "tinker__llama_3_1_8b_instruct",
    "tinker__qwen3_8b",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-script", default="generate_wildbench_tinker_base_repetitions.py")
    parser.add_argument("--generators-file", default="tinker_generators.json")
    parser.add_argument("--out-dir", default="results/wild_bench_v2/passk10")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Debug only. Omit for all 1024 WildBench tasks.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    generator_script = Path(args.generator_script)
    if not generator_script.exists():
        raise FileNotFoundError(f"Missing generator script: {generator_script}")

    for generator in GENERATORS:
        cmd = [
            sys.executable,
            str(generator_script),
            "--generator", generator,
            "--generators-file", args.generators_file,
            "--out-dir", args.out_dir,
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
