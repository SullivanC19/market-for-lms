#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import constants
from openai import OpenAI


def local_desc(batch_file: Path) -> str:
    # WildBench submit_batch.py uses:
    # description = filepath.replace(".batch-submit.jsonl", "")
    return str(batch_file).replace(".batch-submit.jsonl", "")


def submit_batch(python_exe: str, wildbench_dir: Path, batch_file: Path, dry_run: bool) -> None:
    cmd = [
        python_exe,
        str(wildbench_dir / "src" / "openai_batch_eval" / "submit_batch.py"),
        str(batch_file),
    ]
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evals-dir", default=constants.DEFAULT_EVALS_DIR)
    p.add_argument("--wildbench-dir", default=constants.DEFAULT_WILDBENCH_REPO_DIR)
    p.add_argument("--judge-model", default=constants.DEFAULT_JUDGE_MODEL)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--max-submit", type=int, default=constants.DEFAULT_MAX_SUBMIT_BATCHES)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    eval_root = Path(args.evals_dir) / "score.v2" / f"eval={args.judge_model}"
    batch_files = sorted(eval_root.glob("*/*.batch-submit.jsonl"))
    if not batch_files:
        raise SystemExit(f"No batch-submit files found under {eval_root}")

    by_desc = {local_desc(path): path for path in batch_files}

    client = OpenAI()
    visible_batches = []
    after = None
    while True:
        page = (
            client.batches.list(limit=constants.OPENAI_BATCH_LIST_PAGE_DEFAULT, after=after)
            if after
            else client.batches.list(limit=constants.OPENAI_BATCH_LIST_PAGE_DEFAULT)
        )
        visible_batches.extend(page.data)
        if not getattr(page, "has_more", False) or not page.data:
            break
        after = page.data[-1].id

    # OpenAI batch metadata description should equal local_desc(batch_file).
    latest_by_desc = {}
    for b in visible_batches:
        desc = (getattr(b, "metadata", None) or {}).get("description")
        if desc in by_desc:
            old = latest_by_desc.get(desc)
            if old is None or b.created_at > old.created_at:
                latest_by_desc[desc] = b

    to_submit = []
    for desc, path in by_desc.items():
        b = latest_by_desc.get(desc)

        # Never resubmit if there is an active or completed batch for this file.
        if b and b.status in constants.OPENAI_BATCH_ACTIVE_STATUSES:
            print(f"active, skip: {path}  batch={b.id} status={b.status}")
            continue
        if b and b.status == "completed":
            counts = getattr(b, "request_counts", None)
            failed = getattr(counts, "failed", 0) if counts else 0
            total = getattr(counts, "total", 0) if counts else 0
            completed = getattr(counts, "completed", 0) if counts else 0

            if failed > 0:
                print(
                    f"completed with failures, will resubmit: {path} "
                    f"batch={b.id} completed={completed} failed={failed} total={total}"
                )
                to_submit.append(path)
                continue

            print(f"completed cleanly, skip: {path}  batch={b.id}")
            continue

        # Submit if never submitted, or if latest submitted batch is terminal-bad.
        if b is None:
            print(f"not submitted / not visible, will submit: {path}")
            to_submit.append(path)
        elif b.status in constants.OPENAI_BATCH_RESUBMIT_TERMINAL_BAD:
            print(f"{b.status}, will resubmit: {path}  old_batch={b.id}")
            to_submit.append(path)
        else:
            print(f"unknown status, skip: {path}  batch={b.id} status={b.status}")

    if args.max_submit is not None:
        to_submit = to_submit[: args.max_submit]

    print(f"\nSubmitting {len(to_submit)} batch files")
    for path in to_submit:
        submit_batch(args.python, Path(args.wildbench_dir), path, args.dry_run)


if __name__ == "__main__":
    main()