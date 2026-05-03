#!/usr/bin/env python3
"""
Table 1 — Per `rep_XX.json` (under --responses-dir), must be a JSON list; count
dict elements into:

  success, error_timeout, error_billing, error_other, total

Table 2 — Per `rep_XX.batch-submit.jsonl` (under --evals-dir / score.v2 /
eval=<judge>/), same per-line `json.loads` counts as below.

Table 3 — Per `rep_XX.batch_results.jsonl`, aligned with
`05_extract_wildbench_batch_results.format_score_results` (paired
`rep_XX.batch-submit.jsonl` in the same folder):

  line_json_fail     — non-empty line is not valid JSON
  invalid_item       — line parses but top-level value is not an object
  unknown_custom_id  — `custom_id` missing from submit file
  no_choices         — no `response.body.choices`
  judge_parse_fail   — judge `message.content` does not parse (wildbench_judge_parse)
  no_score           — parsed judge dict has no score / Score
  bad_custom_id      — `custom_id` has no `||`
  extract_ok         — row would be written to rep_*.json by extract
  lines_total        — non-empty lines in the batch_results file

Same generator and repetition index columns as table 1.

If a rep JSON file does not parse as a list, table 1 counts are 0 (stderr warning).
Three titled tables are printed to stdout.

Default layouts:
  <responses-dir>/<generator>/rep_XX.json
  <evals-dir>/score.v2/eval=<judge>/<generator>/rep_XX.batch-submit.jsonl
  <evals-dir>/score.v2/eval=<judge>/<generator>/rep_XX.batch_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import constants
from wildbench_judge_parse import parse_judge_json_object, score_value_from_parsed

BATCH_EXTRACT_BUCKETS = (
    "line_json_fail",
    "invalid_item",
    "unknown_custom_id",
    "no_choices",
    "judge_parse_fail",
    "no_score",
    "bad_custom_id",
    "extract_ok",
    "lines_total",
)


def is_timeout_message(msg: str) -> bool:
    m = msg.lower()
    if "timeouterror" in m.replace(" ", ""):
        return True
    if "readtimeout" in m.replace("_", "").lower():
        return True
    if "connecttimeout" in m.replace("_", "").lower():
        return True
    if "asyncio.exceptions.timeouterror" in m:
        return True
    if "timed out" in m:
        return True
    if "wait_for" in m and "timeout" in m:
        return True
    if "deadlineexceeded" in m.replace(" ", ""):
        return True
    return False


def is_billing_402_message(msg: str) -> bool:
    if re.search(r"\b402\b", msg):
        return True
    ml = msg.lower()
    if "payment required" in ml:
        return True
    if "insufficient" in ml and ("fund" in ml or "quota" in ml or "balance" in ml or "credit" in ml):
        return True
    if "billing" in ml and ("disabled" in ml or "block" in ml or "suspend" in ml or "status" in ml):
        return True
    return False


def classify_row(row: dict[str, Any]) -> str:
    err = row.get("error")
    if err is None:
        return "success"
    es = str(err)
    if is_timeout_message(es):
        return "error_timeout"
    if is_billing_402_message(es):
        return "error_billing"
    return "error_other"


def rep_index_from_stem(stem: str) -> int | None:
    if not stem.startswith("rep_"):
        return None
    tail = stem[len("rep_") :]
    if not tail.isdigit():
        return None
    return int(tail)


def iter_rep_json_files(responses_dir: Path, generators: set[str] | None) -> list[tuple[str, int, Path]]:
    out: list[tuple[str, int, Path]] = []
    if not responses_dir.is_dir():
        return out
    for gen_dir in sorted(responses_dir.iterdir()):
        if not gen_dir.is_dir():
            continue
        gen = gen_dir.name
        if generators is not None and gen not in generators:
            continue
        for path in sorted(gen_dir.glob("rep_*.json")):
            ri = rep_index_from_stem(path.stem)
            if ri is None:
                continue
            out.append((gen, ri, path))
    return out


def iter_rep_jsonl_under_eval(
    evals_dir: Path,
    judge_model: str,
    generators: set[str] | None,
    *,
    stem_suffix: str,
    glob_pattern: str,
) -> list[tuple[str, int, Path]]:
    root = evals_dir / "score.v2" / f"eval={judge_model}"
    out: list[tuple[str, int, Path]] = []
    if not root.is_dir():
        return out
    for gen_dir in sorted(root.iterdir()):
        if not gen_dir.is_dir():
            continue
        gen = gen_dir.name
        if generators is not None and gen not in generators:
            continue
        for path in sorted(gen_dir.glob(glob_pattern)):
            stem = path.name.replace(stem_suffix, "")
            ri = rep_index_from_stem(stem)
            if ri is None:
                continue
            out.append((gen, ri, path))
    return out


def iter_batch_submit_jsonl(
    evals_dir: Path, judge_model: str, generators: set[str] | None
) -> list[tuple[str, int, Path]]:
    return iter_rep_jsonl_under_eval(
        evals_dir,
        judge_model,
        generators,
        stem_suffix=".batch-submit.jsonl",
        glob_pattern="rep_*.batch-submit.jsonl",
    )


def iter_batch_results_jsonl(
    evals_dir: Path, judge_model: str, generators: set[str] | None
) -> list[tuple[str, int, Path]]:
    return iter_rep_jsonl_under_eval(
        evals_dir,
        judge_model,
        generators,
        stem_suffix=".batch_results.jsonl",
        glob_pattern="rep_*.batch_results.jsonl",
    )


def load_submission_by_custom_id(submit_path: Path) -> dict[str, Any]:
    """Map custom_id -> submit row; missing file → {} with stderr."""
    by_id: dict[str, Any] = {}
    if not submit_path.is_file():
        print(f"warning: missing batch-submit for extract-style audit: {submit_path}", file=sys.stderr)
        return by_id
    try:
        with submit_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as e:
                    print(f"warning: invalid JSONL in {submit_path}:{line_no}: {e}", file=sys.stderr)
                    continue
                cid = row.get("custom_id")
                if cid is not None:
                    by_id[str(cid)] = row
    except OSError as e:
        print(f"warning: could not read {submit_path}: {e}", file=sys.stderr)
    return by_id


def classify_batch_result_item(
    item: dict[str, Any],
    submission_by_id: dict[str, Any],
) -> str:
    """Same branch order as format_score_results (extract script)."""
    custom_id = item.get("custom_id")
    key = str(custom_id) if custom_id is not None else ""
    submission = submission_by_id.get(key) if key else None
    if submission is None:
        return "unknown_custom_id"

    response = item.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return "no_choices"

    content = choices[0].get("message", {}).get("content", "")
    parsed = parse_judge_json_object(content) if content else None
    if parsed is None:
        return "judge_parse_fail"

    score_val = score_value_from_parsed(parsed)
    if score_val is None:
        return "no_score"

    parts = str(custom_id).split("||")
    if len(parts) < 2:
        return "bad_custom_id"
    return "extract_ok"


def count_batch_results_extract_style(results_path: Path, submit_path: Path) -> dict[str, int]:
    counts = {k: 0 for k in BATCH_EXTRACT_BUCKETS}
    submission_by_id = load_submission_by_custom_id(submit_path)
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            counts["lines_total"] += 1
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                counts["line_json_fail"] += 1
                continue
            if not isinstance(item, dict):
                counts["invalid_item"] += 1
                continue
            counts[classify_batch_result_item(item, submission_by_id)] += 1
    return counts


def collect_batch_results_failed_samples(
    results_path: Path, submit_path: Path, limit: int
) -> list[tuple[dict[str, Any], str]]:
    """Return list of (item, classification) for failed items up to limit."""
    failed_items: list[tuple[dict[str, Any], str]] = []
    submission_by_id = load_submission_by_custom_id(submit_path)
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                failed_items.append((item, "line_json_fail"))
                if len(failed_items) >= limit:
                    break
                continue
            if not isinstance(item, dict):
                failed_items.append((item, "invalid_item"))
                if len(failed_items) >= limit:
                    break
                continue
            classification = classify_batch_result_item(item, submission_by_id)
            if classification != "extract_ok":
                failed_items.append((item, classification))
                if len(failed_items) >= limit:
                    break
    return failed_items


def count_jsonl_parse_results(path: Path) -> tuple[int, int, int]:
    """Return (lines_ok, lines_fail, lines_total) for non-empty lines only."""
    ok = fail = 0
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
            try:
                json.loads(stripped)
                ok += 1
            except json.JSONDecodeError:
                fail += 1
    return ok, fail, total


def print_table(title: str, cols: list[str], rows: list[list[str]]) -> None:
    print()
    print(title)
    widths = [len(h) for h in cols]
    for cells in rows:
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(fmt(list(cols)))
    print(fmt(["-" * w for w in widths]))
    for cells in rows:
        print(fmt(cells))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--responses-dir", type=Path, default=Path(constants.DEFAULT_RESPONSES_DIR))
    p.add_argument("--evals-dir", type=Path, default=Path(constants.DEFAULT_EVALS_DIR))
    p.add_argument("--judge-model", default=constants.DEFAULT_JUDGE_MODEL)
    p.add_argument(
        "--generators",
        nargs="*",
        default=None,
        help="If set, only these generator directory names.",
    )
    p.add_argument(
        "--show-failed-samples",
        type=int,
        default=0,
        help="Number of failed samples to print from response files (0 = don't show).",
    )
    args = p.parse_args()

    responses_dir = args.responses_dir
    evals_dir = args.evals_dir
    gen_filter = set(args.generators) if args.generators else None

    # --- Table 1: rep_*.json ---
    rep_files = iter_rep_json_files(responses_dir, gen_filter)
    if not rep_files:
        raise SystemExit(f"No rep_*.json under {responses_dir}")

    cols1 = ["generator", "rep", "success", "error_timeout", "error_billing", "error_other", "total"]
    rows1: list[list[str]] = []

    for gen, rep, path in rep_files:
        try:
            raw = path.read_text(encoding="utf-8")
            val = json.loads(raw)
        except (json.JSONDecodeError, OSError) as e:
            print(f"warning: not valid JSON list, skipping row counts: {path}: {e}", file=sys.stderr)
            rows1.append([gen, str(rep), "0", "0", "0", "0", "0"])
            continue

        if not isinstance(val, list):
            print(f"warning: top-level JSON is not a list: {path}", file=sys.stderr)
            rows1.append([gen, str(rep), "0", "0", "0", "0", "0"])
            continue

        c = defaultdict(int)
        for item in val:
            if not isinstance(item, dict):
                continue
            classification = classify_row(item)
            c[classification] += 1
        tot = sum(c.values())
        rows1.append(
            [
                gen,
                str(rep),
                str(c["success"]),
                str(c["error_timeout"]),
                str(c["error_billing"]),
                str(c["error_other"]),
                str(tot),
            ]
        )

    print_table("Responses (rep_*.json)", cols1, rows1)

    # --- Table 2: rep_*.batch-submit.jsonl ---
    cols2 = ["generator", "rep", "lines_json_ok", "lines_json_fail", "lines_total"]
    rows2: list[list[str]] = []

    batch_files = iter_batch_submit_jsonl(evals_dir, args.judge_model, gen_filter)
    if not batch_files:
        submit_root = evals_dir / "score.v2" / f"eval={args.judge_model}"
        print(f"\n(no rep_*.batch-submit.jsonl under {submit_root})", file=sys.stderr)
    for gen, rep, path in batch_files:
        try:
            ok, fail, tot = count_jsonl_parse_results(path)
        except OSError as e:
            print(f"warning: could not read {path}: {e}", file=sys.stderr)
            rows2.append([gen, str(rep), "0", "0", "0"])
            continue
        rows2.append([gen, str(rep), str(ok), str(fail), str(tot)])

    print_table("Batch submit JSONL (json.loads per non-empty line)", cols2, rows2)

    # --- Table 3: rep_*.batch_results.jsonl (extract-style) ---
    cols3 = ["generator", "rep"] + [b for b in BATCH_EXTRACT_BUCKETS]
    rows3: list[list[str]] = []
    failed_samples: list[tuple[str, int, dict[str, Any], str]] = []  # (gen, rep, item, classification)

    results_files = iter_batch_results_jsonl(evals_dir, args.judge_model, gen_filter)
    if not results_files:
        results_root = evals_dir / "score.v2" / f"eval={args.judge_model}"
        print(f"\n(no rep_*.batch_results.jsonl under {results_root})", file=sys.stderr)
    for gen, rep, path in results_files:
        submit_path = path.with_name(path.name.replace("batch_results.jsonl", "batch-submit.jsonl"))
        try:
            c = count_batch_results_extract_style(path, submit_path)
            # Collect failed samples if requested
            if args.show_failed_samples > 0:
                failed_items = collect_batch_results_failed_samples(path, submit_path, args.show_failed_samples)
                for item, classification in failed_items:
                    failed_samples.append((gen, rep, item, classification))
        except OSError as e:
            print(f"warning: could not read {path}: {e}", file=sys.stderr)
            rows3.append([gen, str(rep)] + ["0"] * len(BATCH_EXTRACT_BUCKETS))
            continue
        rows3.append([gen, str(rep)] + [str(c[b]) for b in BATCH_EXTRACT_BUCKETS])

    print_table("Batch results JSONL (extract-style row classification)", cols3, rows3)

    # --- Print failed samples if requested ---
    if args.show_failed_samples > 0 and failed_samples:
        print()
        print(f"Failed samples (showing up to {args.show_failed_samples}):")
        print()
        for i, (gen, rep, item, classification) in enumerate(failed_samples[: args.show_failed_samples]):
            print(f"[{i + 1}] Generator: {gen}, Rep: {rep}, Classification: {classification}")
            print(json.dumps(item, indent=2))
            print()


if __name__ == "__main__":
    main()
