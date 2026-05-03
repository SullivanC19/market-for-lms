#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from json import JSONDecodeError
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import constants


def patch_file(path: Path) -> int:
    changed = 0
    out_lines: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSONL in {path}:{line_no}\n"
                    f"{e}\n"
                    f"snippet={line[:500]!r}"
                ) from e

            body = obj.get("body", {})

            if "max_tokens" in body:
                body["max_completion_tokens"] = body.pop("max_tokens")
                changed += 1

            for key in constants.GPT5_BATCH_SUBMIT_BODY_DROP_KEYS:
                if key in body:
                    body.pop(key)
                    changed += 1

            obj["body"] = body
            out_lines.append(json.dumps(obj, ensure_ascii=False))

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    return changed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evals-dir", default=constants.DEFAULT_EVALS_DIR)
    p.add_argument("--judge-model", default=constants.DEFAULT_JUDGE_MODEL)
    args = p.parse_args()

    root = Path(args.evals_dir) / "score.v2" / f"eval={args.judge_model}"
    files = sorted(root.glob("*/*.batch-submit.jsonl"))

    if not files:
        raise SystemExit(f"No batch-submit files found under {root}")

    print(f"Found {len(files)} batch-submit files under {root}")

    total = 0
    for path in files:
        n = patch_file(path)
        total += n
        print(f"{path}: patched {n}")

    print(f"Patched {total} fields across {len(files)} files")


if __name__ == "__main__":
    main()