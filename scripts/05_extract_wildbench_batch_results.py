#!/usr/bin/env python3
"""
Extract completed OpenAI Batch results for WildBench score evals.

Expected input files:
  eval_results/passk10/score.v2/eval=<judge>/<generator>/rep_00.batch-submit.jsonl

Expected output files:
  eval_results/passk10/score.v2/eval=<judge>/<generator>/rep_00.batch_results.jsonl
  eval_results/passk10/score.v2/eval=<judge>/<generator>/rep_00.json

This script finds OpenAI batches by the metadata description written by
WildBench's submit_batch.py:
  description = <batch_submit_path without .batch-submit.jsonl>

It then downloads completed batch output to rep_*.batch_results.jsonl and
writes per-task rep_*.json for other pipelines. summarize_wildbench_scores.py
reads the .batch_results.jsonl files directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import constants
from wildbench_judge_parse import (
    coerce_judge_message_content,
    parse_judge_json_object,
    score_value_from_parsed,
)

from openai import OpenAI


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL in {path}:{line_no}: {e}") from e
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def load_generators(path: Path, use_all: bool) -> list[str]:
    if not use_all:
        return list(constants.DEFAULT_GENERATORS)
    payload = json.loads(path.read_text())
    generators = [g["generator"] for g in payload.get("generators", [])]
    if not generators:
        raise ValueError(f"No generators found in {path}")
    return generators


def iter_batch_submit_files(
    *,
    evals_dir: Path,
    judge_model: str,
    generators: Iterable[str],
) -> list[Path]:
    root = evals_dir / "score.v2" / f"eval={judge_model}"
    files: list[Path] = []
    for generator in generators:
        gen_dir = root / generator
        if not gen_dir.exists():
            print(f"missing eval generator dir: {gen_dir}")
            continue
        files.extend(sorted(gen_dir.glob("rep_*.batch-submit.jsonl")))
    return files


def batch_description_for_submit_file(path: Path) -> str:
    return str(path).replace(".batch-submit.jsonl", "")


def list_batches_by_description(client: OpenAI, *, limit: int) -> dict[str, list[Any]]:
    """Return {metadata.description: [batch, ...]}, newest first when possible."""
    by_desc: dict[str, list[Any]] = {}

    page_limit = min(max(limit, 1), constants.OPENAI_BATCHES_LIST_PAGE_LIMIT_MAX)

    # The OpenAI Python SDK returns a SyncCursorPage; iteration handles pagination.
    count = 0
    for batch in client.batches.list(limit=page_limit):
        count += 1
        meta = getattr(batch, "metadata", None) or {}
        desc = meta.get("description")
        if not desc:
            continue
        by_desc.setdefault(desc, []).append(batch)

    for batches in by_desc.values():
        batches.sort(key=lambda b: getattr(b, "created_at", 0) or 0, reverse=True)

    print(f"indexed {count} recent OpenAI batches (page_limit={page_limit}, requested={limit})")
    return by_desc


def choose_batch(batches: list[Any]) -> Any:
    completed = [b for b in batches if getattr(b, "status", None) == "completed"]
    if completed:
        return completed[0]
    return batches[0]


def download_file_text(client: OpenAI, file_id: str) -> str:
    content = client.files.content(file_id)

    # Different SDK versions expose content differently. Support the common ones.
    if hasattr(content, "text"):
        text_attr = content.text
        if callable(text_attr):
            return text_attr()
        return str(text_attr)

    if hasattr(content, "read"):
        data = content.read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)

    if hasattr(content, "content"):
        data = content.content
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)

    # Last-resort path: the SDK object supports write_to_file reliably.
    tmp = Path(".openai_batch_download_tmp.jsonl")
    try:
        content.write_to_file(tmp)
        return tmp.read_text()
    finally:
        if tmp.exists():
            tmp.unlink()


def extract_model_output_from_prompt(prompt: str) -> str | None:
    start = "<|begin_of_response|>\n"
    end = "<|end_of_response|>\n"
    if start not in prompt:
        return None
    tail = prompt.split(start, 1)[1]
    if end in tail:
        return tail.split(end, 1)[0].strip()
    return tail.strip()


def format_score_results(
    *,
    submit_file: Path,
    batch_results_file: Path,
    out_json_file: Path,
    verbose_skips: int = 0,
) -> int:
    submissions = read_jsonl(submit_file)
    results = read_jsonl(batch_results_file)
    submission_by_id = {row["custom_id"]: row for row in submissions}

    formatted: list[dict[str, Any]] = []
    skipped = 0
    skips_logged = 0

    def log_skip(reason: str, cid: Any, extra: str = "") -> None:
        nonlocal skips_logged
        if verbose_skips <= 0 or skips_logged >= verbose_skips:
            return
        msg = f"skip {reason} custom_id={cid!r}"
        if extra:
            msg += f" {extra}"
        print(msg, file=sys.stderr)
        skips_logged += 1

    for item in results:
        custom_id = item.get("custom_id")
        submission = submission_by_id.get(custom_id)
        if submission is None:
            skipped += 1
            log_skip("unknown_custom_id", custom_id)
            continue

        response = item.get("response") or {}
        body = response.get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            skipped += 1
            log_skip("no_choices", custom_id, f"status_code={response.get('status_code')}")
            continue

        content = choices[0].get("message", {}).get("content", "")
        parsed = parse_judge_json_object(content) if content else None
        if parsed is None:
            skipped += 1
            flat = coerce_judge_message_content(content)
            preview = (flat[:120] + "…") if len(flat) > 120 else flat
            preview = preview.replace("\n", "\\n")
            log_skip("json_parse", custom_id, f"content_preview={preview!r}")
            continue

        score_val = score_value_from_parsed(parsed)
        if score_val is None:
            skipped += 1
            log_skip("no_score", custom_id, f"keys={list(parsed.keys())}")
            continue

        parts = custom_id.split("||")
        if len(parts) < 2:
            skipped += 1
            log_skip("bad_custom_id", custom_id)
            continue

        session_id = parts[0]
        model_test = parts[1]
        prompt = submission.get("body", {}).get("messages", [{}])[0].get("content", "")

        formatted.append(
            {
                "session_id": session_id,
                "parsed_result": parsed,
                "meta_data": {
                    "batch_req_id": item.get("id"),
                    "usage": body.get("usage"),
                    "error": item.get("error"),
                },
                "model_test": model_test,
                "score": score_val,
                "model_output": extract_model_output_from_prompt(prompt),
            }
        )

    # OpenAI batch output order is not guaranteed. Preserve benchmark order by
    # ordering according to the original submit file custom_id order.
    order = {row["custom_id"].split("||")[0]: i for i, row in enumerate(submissions)}
    formatted.sort(key=lambda row: order.get(row["session_id"], 10**12))

    write_json(out_json_file, formatted)
    return skipped


def process_once(args: argparse.Namespace) -> bool:
    client = OpenAI()
    generators = args.generators or load_generators(Path(args.generators_file), args.all_generators_in_registry)
    submit_files = iter_batch_submit_files(
        evals_dir=Path(args.evals_dir),
        judge_model=args.judge_model,
        generators=generators,
    )
    if not submit_files:
        raise SystemExit("No .batch-submit.jsonl files found.")

    by_desc = list_batches_by_description(client, limit=args.batch_list_limit)

    statuses: dict[str, int] = {}
    made_progress = False

    for submit_file in submit_files:
        out_json = submit_file.with_suffix("").with_suffix(".json")
        batch_results = submit_file.with_name(submit_file.name.replace(".batch-submit.jsonl", ".batch_results.jsonl"))
        desc = batch_description_for_submit_file(submit_file)

        if out_json.exists() and not args.overwrite:
            statuses["already_extracted"] = statuses.get("already_extracted", 0) + 1
            continue

        matches = by_desc.get(desc, [])
        if not matches:
            print(f"not submitted or not found: {submit_file}")
            statuses["not_found"] = statuses.get("not_found", 0) + 1
            continue

        batch = choose_batch(matches)
        status = getattr(batch, "status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        print(f"{status:>12} {batch.id} {desc}")

        if status in constants.OPENAI_BATCH_TERMINAL_FAILURE_STATUSES:
            err = getattr(batch, "errors", None)
            if err:
                print(f"  errors: {err}")
            continue

        if status != "completed":
            continue

        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            print(f"  completed but missing output_file_id: {batch.id}")
            continue

        if not batch_results.exists() or args.overwrite:
            text = download_file_text(client, output_file_id)
            write_text(batch_results, text)
            print(f"  wrote {batch_results}")
            made_progress = True

        skipped = format_score_results(
            submit_file=submit_file,
            batch_results_file=batch_results,
            out_json_file=out_json,
            verbose_skips=args.verbose_skips,
        )
        print(f"  wrote {out_json} (skipped malformed/unparseable rows: {skipped})")
        made_progress = True

    print("\nstatus counts:")
    for key in sorted(statuses):
        print(f"  {key}: {statuses[key]}")

    remaining = sum(v for k, v in statuses.items() if k not in {"already_extracted", "completed"})
    return made_progress, remaining


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evals-dir", default=constants.DEFAULT_EVALS_DIR)
    parser.add_argument("--generators-file", default=constants.DEFAULT_GENERATORS_REGISTRY)
    parser.add_argument("--generators", nargs="*", default=None)
    parser.add_argument("--all-generators-in-registry", action="store_true")
    parser.add_argument("--judge-model", default=constants.DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--batch-list-limit",
        type=int,
        default=constants.OPENAI_BATCH_LIST_PAGE_DEFAULT,
        help=(
            "Page size when listing batches (capped at "
            f"{constants.OPENAI_BATCHES_LIST_PAGE_LIMIT_MAX} per OpenAI API; SDK pagination still walks all pages)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verbose-skips",
        type=int,
        default=0,
        metavar="N",
        help="Log up to N skipped rows (to stderr) with reason, e.g. json_parse or no_score.",
    )
    args = parser.parse_args()

    made_progress, remaining = process_once(args)
    if remaining == 0:
        print("All batches extracted.")
        return
    if not made_progress:
        print("No progress made.")
    if remaining > 0:
        print(f"\n{remaining} batches remain; run again to extract more.")


if __name__ == "__main__":
    main()
