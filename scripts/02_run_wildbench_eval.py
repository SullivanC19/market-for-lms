#!/usr/bin/env python3
"""
Create WildBench judge batch files for generated local-result JSONs.

Expected generated-output layout:

  results/responses/passk10/<generator>/rep_00.json
  results/responses/passk10/<generator>/rep_01.json
  ...

For each rep_XX.json, this runs WildBench's src/eval.py in score mode and writes:

  results/evals/passk10/score.v2/eval=<judge>/<generator>/rep_XX.batch-submit.jsonl

By default, this only prepares OpenAI Batch input files. Pass --submit to call
WildBench's src/openai_batch_eval/submit_batch.py on each generated file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import constants


def load_generators_from_registry(path: Path, keep_defaults_only: bool) -> list[str]:
    if keep_defaults_only:
        return list(constants.DEFAULT_GENERATORS)

    payload = json.loads(path.read_text())
    generators = [g["generator"] for g in payload.get("generators", [])]
    if not generators:
        raise ValueError(f"No generators found in {path}")
    return generators


def existing_rep_files(outputs_dir: Path, generator: str) -> list[Path]:
    gen_dir = outputs_dir / generator
    if not gen_dir.exists():
        return []
    return sorted(gen_dir.glob("rep_*.json"))


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_eval_cmd(
    *,
    python_exe: str,
    wildbench_dir: Path,
    judge_model: str,
    eval_template: Path,
    target_model_name: str,
    local_result_file: Path,
    eval_output_file: Path,
    max_words_to_eval: int | None,
) -> list[str]:
    cmd = [
        python_exe,
        str(wildbench_dir / "src" / "eval.py"),
        "--batch_mode",
        "--action",
        "eval",
        "--mode",
        "score",
        "--model",
        judge_model,
        "--eval_template",
        str(eval_template),
        "--target_model_name",
        target_model_name,
        "--local_result_file",
        str(local_result_file),
        "--eval_output_file",
        str(eval_output_file),
    ]
    if max_words_to_eval is not None:
        cmd.extend(["--max_words_to_eval", str(max_words_to_eval)])
    return cmd


def build_submit_cmd(*, python_exe: str, wildbench_dir: Path, batch_file: Path) -> list[str]:
    return [
        python_exe,
        str(wildbench_dir / "src" / "openai_batch_eval" / "submit_batch.py"),
        str(batch_file),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-dir", default=constants.DEFAULT_RESPONSES_DIR)
    parser.add_argument("--evals-dir", default=constants.DEFAULT_EVALS_DIR)
    parser.add_argument("--wildbench-dir", default=constants.DEFAULT_WILDBENCH_REPO_DIR, help="Path to cloned WildBench repo root")
    parser.add_argument("--generators-file", default=constants.DEFAULT_GENERATORS_REGISTRY)
    parser.add_argument(
        "--generators",
        nargs="*",
        default=None,
        help="Optional explicit generator keys. If omitted, uses the top-7 defaults.",
    )
    parser.add_argument(
        "--all-generators-in-registry",
        action="store_true",
        help="Use every generator in --generators-file instead of the top-7 defaults.",
    )
    parser.add_argument("--judge-model", default=constants.DEFAULT_JUDGE_MODEL)
    parser.add_argument("--eval-template", default=constants.DEFAULT_EVAL_TEMPLATE_RELATIVE)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--max-words-to-eval",
        type=int,
        default=None,
        help="Optional override for WildBench eval.py --max_words_to_eval. Omit to use eval.py default.",
    )
    parser.add_argument("--submit", action="store_true", help="Submit each batch file after creating it")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    responses_dir = Path(args.responses_dir)
    evals_dir = Path(args.evals_dir)
    wildbench_dir = Path(args.wildbench_dir)
    generators_file = Path(args.generators_file)
    eval_template = Path(args.eval_template)
    if not eval_template.is_absolute():
        eval_template = wildbench_dir / eval_template

    if args.generators:
        generators = args.generators
    else:
        generators = load_generators_from_registry(
            generators_file,
            keep_defaults_only=not args.all_generators_in_registry,
        )

    jobs: list[tuple[str, Path, Path]] = []
    for generator in generators:
        rep_files = existing_rep_files(responses_dir, generator)
        if not rep_files:
            print(f"Skipping {generator}: no rep_*.json files under {responses_dir / generator}")
            continue

        for rep_file in rep_files:
            rep_name = rep_file.stem  # rep_00
            out_file = (
                evals_dir
                / "score.v2"
                / f"eval={args.judge_model}"
                / generator
                / f"{rep_name}.batch-submit.jsonl"
            )
            jobs.append((generator, rep_file, out_file))

    if not jobs:
        raise SystemExit("No eval jobs found.")

    print(f"Found {len(jobs)} eval jobs")
    print(f"Judge model: {args.judge_model}")

    created: list[Path] = []
    for generator, rep_file, out_file in jobs:
        if out_file.exists() and not args.overwrite:
            print(f"Exists, skipping: {out_file}")
            created.append(out_file)
            continue

        out_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_eval_cmd(
            python_exe=args.python,
            wildbench_dir=wildbench_dir,
            judge_model=args.judge_model,
            eval_template=eval_template,
            target_model_name=generator,
            local_result_file=rep_file,
            eval_output_file=out_file,
            max_words_to_eval=args.max_words_to_eval,
        )
        run_cmd(cmd, dry_run=args.dry_run)
        created.append(out_file)

    if args.submit:
        for batch_file in created:
            if not batch_file.exists() and not args.dry_run:
                print(f"Skipping submit; missing batch file: {batch_file}")
                continue
            cmd = build_submit_cmd(
                python_exe=args.python,
                wildbench_dir=wildbench_dir,
                batch_file=batch_file,
            )
            run_cmd(cmd, dry_run=args.dry_run)

    print("Done")


if __name__ == "__main__":
    main()
