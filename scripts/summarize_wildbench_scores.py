#!/usr/bin/env python3
"""Summarize WildBench scores (from batch_results JSONL) and generation costs (from rep JSON)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import constants
from wildbench_judge_parse import parse_judge_json_object, score_value_from_parsed

import numpy as np
import pandas as pd


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def parse_score_from_batch_line(item: dict) -> tuple[str, float] | None:
    """Return (session_id, raw_score) from one batch_results.jsonl row, or None."""
    custom_id = item.get("custom_id")
    if not custom_id or "||" not in str(custom_id):
        return None
    session_id = str(custom_id).split("||", 1)[0]

    response = item.get("response") or {}
    if response.get("status_code") != 200:
        return None

    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None

    content = choices[0].get("message", {}).get("content", "")
    parsed = parse_judge_json_object(content)
    if parsed is None:
        return None

    score = score_value_from_parsed(parsed)
    if score is None:
        return None

    try:
        raw = float(score)
    except (TypeError, ValueError):
        return None

    return session_id, raw


def iter_jsonl_rows(path: Path):
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"warning: skip invalid JSON in {path}:{line_no}: {e}", file=sys.stderr)


def scores_map_from_batch_results(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in iter_jsonl_rows(path):
        out = parse_score_from_batch_line(item)
        if out is None:
            continue
        sid, raw = out
        scores[sid] = raw
    return scores


def ordered_session_ids_from_outputs(outputs: list) -> list[str]:
    """Stable item axis: row_index when present, else file order."""
    if not outputs:
        return []
    if all("row_index" in r and r.get("row_index") is not None for r in outputs):
        ordered = sorted(outputs, key=lambda r: (int(r["row_index"]), str(r["session_id"])))
    else:
        ordered = outputs
    return [str(r["session_id"]) for r in ordered]


def outputs_sid_index(outputs: list) -> dict[str, dict]:
    return {str(r["session_id"]): r for r in outputs}


def union_rep_stems(eval_root: Path, model_names: list[str]) -> list[str]:
    stems: set[str] = set()
    for g in model_names:
        for p in (eval_root / g).glob("rep_*.batch_results.jsonl"):
            stems.add(p.name.removesuffix(".batch_results.jsonl"))
    return sorted(stems)


def reference_task_order(
    responses_dir: Path, model_names: list[str], reps: list[str]
) -> tuple[list[str], str, str]:
    """Pick reference (generator, rep) outputs to define the item axis."""
    for rep in sorted(reps):
        for g in sorted(model_names):
            p = responses_dir / g / f"{rep}.json"
            if p.exists():
                outputs = load_json(p)
                if isinstance(outputs, list) and outputs:
                    return ordered_session_ids_from_outputs(outputs), g, rep
    raise SystemExit("Could not find any rep_*.json under responses-dir to define task order.")


def save_model_item_rep_matrices(
    *,
    matrices_out: Path,
    eval_root: Path,
    responses_dir: Path,
    model_names: list[str],
    session_ids: list[str],
    reps: list[str],
) -> None:
    """
    Write compressed .npz with:
      success (float64): 1 = parsed judge raw score > 5, 0 = score present and <= 5 or missing judge
        for that task/rep, nan = no generation row for that cell
      cost_usd (float64): naive estimated_cost_usd per generation row; nan = missing
    Axes order (model, item, repetition) = (generator, task index, rep stem index).
    """
    n_m, n_t, n_r = len(model_names), len(session_ids), len(reps)
    success = np.full((n_m, n_t, n_r), np.nan, dtype=np.float64)
    cost_usd = np.full((n_m, n_t, n_r), np.nan, dtype=np.float64)

    sid_to_item = {sid: i for i, sid in enumerate(session_ids)}

    for mi, gen in enumerate(model_names):
        for ri, rep in enumerate(reps):
            out_path = responses_dir / gen / f"{rep}.json"
            batch_path = eval_root / gen / f"{rep}.batch_results.jsonl"
            if not out_path.exists() or not batch_path.exists():
                continue
            outputs = load_json(out_path)
            if not isinstance(outputs, list):
                continue
            by_sid = outputs_sid_index(outputs)
            score_by_sid = scores_map_from_batch_results(batch_path)

            for sid in session_ids:
                ti = sid_to_item[sid]
                row = by_sid.get(sid)
                if row is None:
                    continue
                raw = score_by_sid.get(sid)
                if raw is None:
                    success[mi, ti, ri] = 0.0
                else:
                    success[mi, ti, ri] = 1.0 if raw > 5.0 else 0.0
                cost_usd[mi, ti, ri] = float(cost_parts(row)["estimated_cost_usd"])

    matrices_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        matrices_out,
        success=success,
        cost_usd=cost_usd,
        model_names=np.array(model_names, dtype=object),
        session_ids=np.array(session_ids, dtype=object),
        reps=np.array(reps, dtype=object),
    )
    print(f"wrote matrices {matrices_out} shape success={success.shape}")


def cost_parts(output_row: dict) -> dict:
    est = output_row.get("price_estimate") or {}
    return {
        "prompt_tokens": est.get("prompt_tokens", 0) or 0,
        "completion_tokens": est.get("completion_tokens", 0) or 0,
        "prefill_cost_usd": est.get("prefill_cost_usd", 0.0) or 0.0,
        "sample_cost_usd": est.get("sample_cost_usd", 0.0) or 0.0,
        "estimated_cost_usd": est.get("estimated_cost_usd", 0.0) or 0.0,
    }


def summarize_generator(
    *,
    generator: str,
    eval_generator_dir: Path,
    outputs_generator_dir: Path,
    failure_raw_score: float,
) -> dict | None:
    score_rows: list[dict] = []
    cost_rows: list[dict] = []

    batch_files = sorted(eval_generator_dir.glob("rep_*.batch_results.jsonl"))
    if not batch_files:
        return None

    for bpath in batch_files:
        rep = bpath.name.removesuffix(".batch_results.jsonl")
        out_path = outputs_generator_dir / f"{rep}.json"
        if not out_path.exists():
            print(f"missing outputs for {generator} {rep}: {out_path}", file=sys.stderr)
            continue

        outputs = load_json(out_path)
        if not isinstance(outputs, list):
            raise ValueError(f"Expected list in {out_path}")

        score_by_sid = scores_map_from_batch_results(bpath)
        output_sids = [str(r["session_id"]) for r in outputs]
        sid_set = set(output_sids)
        missing = [sid for sid in output_sids if sid not in score_by_sid]
        extra = [sid for sid in score_by_sid if sid not in sid_set]
        if missing or extra:
            print(
                f"warning {bpath}: outputs={len(outputs)} unique_scores={len(score_by_sid)} "
                f"missing_judge={len(missing)} batch_only={len(extra)}; "
                f"missing imputed as raw={failure_raw_score}",
                file=sys.stderr,
            )

        for out_row in outputs:
            sid = str(out_row["session_id"])
            raw = score_by_sid.get(sid, failure_raw_score)
            score_rows.append(
                {
                    "generator": generator,
                    "rep": rep,
                    "session_id": sid,
                    "raw_score": raw,
                    "wb_score": (raw - 5.0) * 2.0,
                }
            )
            c = cost_parts(out_row)
            cost_rows.append(
                {
                    "generator": generator,
                    "rep": rep,
                    "row_index": out_row.get("row_index"),
                    "session_id": sid,
                    **c,
                }
            )

    if not score_rows:
        return None

    scores = pd.DataFrame(score_rows)
    costs = pd.DataFrame(cost_rows)

    request_costs = (
        costs.groupby("row_index", dropna=False)
        .agg(
            prompt_tokens=("prompt_tokens", "first"),
            completion_tokens=("completion_tokens", "sum"),
            prefill_cost_usd=("prefill_cost_usd", "first"),
            sample_cost_usd=("sample_cost_usd", "sum"),
        )
        .reset_index()
    )
    request_costs["estimated_cost_usd"] = (
        request_costs["prefill_cost_usd"] + request_costs["sample_cost_usd"]
    )

    return {
        "generator": generator,
        "n_evals": len(scores),
        "n_tasks": scores["session_id"].nunique(),
        "n_reps": scores["rep"].nunique(),
        "avg_raw_score": scores["raw_score"].mean(),
        "avg_wb_score": scores["wb_score"].mean(),
        "actual_estimated_generation_cost_usd": request_costs["estimated_cost_usd"].sum(),
        "naive_per_sample_cost_usd": costs["estimated_cost_usd"].sum(),
        "avg_actual_cost_per_task_usd": request_costs["estimated_cost_usd"].mean(),
        "avg_naive_cost_per_sample_usd": costs["estimated_cost_usd"].mean(),
        "total_prompt_tokens_actual": request_costs["prompt_tokens"].sum(),
        "total_completion_tokens": request_costs["completion_tokens"].sum(),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Scores from rep_*.batch_results.jsonl; costs from rep_*.json under outputs-dir."
    )
    p.add_argument(
        "--evals-dir",
        type=Path,
        default=Path(constants.DEFAULT_EVALS_DIR),
        help="Contains score.v2/eval=<judge-model>/<generator>/rep_*.batch_results.jsonl",
    )
    p.add_argument(
        "--responses-dir",
        type=Path,
        default=Path(constants.DEFAULT_RESPONSES_DIR),
        help="Contains <generator>/rep_*.json generation logs (price_estimate).",
    )
    p.add_argument("--judge-model", default=constants.DEFAULT_JUDGE_MODEL)
    p.add_argument(
        "--failure-raw-score",
        type=float,
        default=constants.DEFAULT_FAILURE_RAW_SCORE,
        help="Raw score (1–10 WildBench scale) imputed when a task has outputs but no judge score.",
    )
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument(
        "--matrices-out",
        type=Path,
        default=None,
        help="Write model×item×rep .npz (success=raw judge score>5, cost_usd, axis labels). "
        "If omitted but --out-csv is set, writes <out-csv stem>_matrices.npz beside the CSV.",
    )
    args = p.parse_args()

    eval_root = args.evals_dir / "score.v2" / f"eval={args.judge_model}"
    if not eval_root.is_dir():
        raise SystemExit(f"missing eval root: {eval_root}")

    rows = []
    matrix_models: list[str] = []
    for gen_dir in sorted(eval_root.iterdir()):
        if not gen_dir.is_dir():
            continue
        out_gen_dir = args.responses_dir / gen_dir.name
        if not out_gen_dir.is_dir():
            print(f"missing outputs dir: {out_gen_dir}", file=sys.stderr)
            continue

        summary = summarize_generator(
            generator=gen_dir.name,
            eval_generator_dir=gen_dir,
            outputs_generator_dir=out_gen_dir,
            failure_raw_score=args.failure_raw_score,
        )
        if summary:
            rows.append(summary)
            matrix_models.append(gen_dir.name)
        else:
            print(f"skip (no rep_*.batch_results.jsonl): {gen_dir}", file=sys.stderr)

    if not rows:
        raise SystemExit("No summaries produced.")

    df = pd.DataFrame(rows).sort_values("avg_wb_score", ascending=False)

    cols = [
        "generator",
        "n_tasks",
        "n_reps",
        "n_evals",
        "avg_raw_score",
        "avg_wb_score",
        "actual_estimated_generation_cost_usd",
        "naive_per_sample_cost_usd",
        "avg_actual_cost_per_task_usd",
        "avg_naive_cost_per_sample_usd",
        "total_prompt_tokens_actual",
        "total_completion_tokens",
    ]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print(df[cols].to_string(index=False))

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df[cols].to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv}")

    matrices_path = args.matrices_out
    if matrices_path is None and args.out_csv is not None:
        matrices_path = args.out_csv.parent / f"{args.out_csv.stem}_matrices.npz"

    if matrices_path is not None:
        model_names = sorted(matrix_models)
        reps = union_rep_stems(eval_root, model_names)
        session_ids, ref_g, ref_r = reference_task_order(args.responses_dir, model_names, reps)
        print(
            f"matrix task axis from reference {ref_g}/{ref_r}.json "
            f"({len(session_ids)} tasks, {len(reps)} reps, {len(model_names)} models)"
        )
        save_model_item_rep_matrices(
            matrices_out=matrices_path,
            eval_root=eval_root,
            responses_dir=args.responses_dir,
            model_names=model_names,
            session_ids=session_ids,
            reps=reps,
        )


if __name__ == "__main__":
    main()
