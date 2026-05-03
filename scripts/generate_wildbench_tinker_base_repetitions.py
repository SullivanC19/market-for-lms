# generate_wildbench_tinker_base_repetitions.py
#
# Generate WildBench local-result JSON files using Tinker base-model inference.
# For each WildBench prompt, the script makes one Tinker call with
# num_samples=<repetitions>, then writes one WildBench-compatible file per
# repetition:
#
#   <responses-dir>/<generator>/rep_00.json
#   <responses-dir>/<generator>/rep_01.json
#   ...
#
# Each rep file is compatible with WildBench src/eval.py:
#   [{"session_id": ..., "generator": ..., "output": ["..."]}, ...]

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import constants
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tqdm import tqdm


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp.replace(path)


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    return payload


def load_generator_config(path: Path, generator: str) -> dict[str, Any]:
    registry = json.loads(path.read_text())
    generators = {item["generator"]: item for item in registry.get("generators", [])}
    if generator not in generators:
        valid = "\n".join(sorted(generators))
        raise ValueError(f"Unknown generator: {generator!r}\n\nValid generators:\n{valid}")
    return generators[generator]


def load_pricing(generator_cfg: dict[str, Any]) -> dict[str, float]:
    pricing = generator_cfg.get("pricing_usd_per_mtok")
    if not isinstance(pricing, dict):
        raise ValueError("Generator config is missing pricing_usd_per_mtok")
    return {"prefill": float(pricing["prefill"]), "sample": float(pricing["sample"])}


def normalize_messages(conversation_input: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in conversation_input:
        role = item.get("role") or item.get("from")
        content = item.get("content", item.get("value", ""))

        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "assistant", "model"}:
            role = "assistant"
        elif role != "system":
            role = "user"

        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        messages.append({"role": role, "content": content})
    return messages


def message_content_to_text(parsed_message: Any) -> str:
    if not isinstance(parsed_message, dict):
        return str(parsed_message)

    content = parsed_message.get("content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return str(content)


def token_count(prompt: Any) -> int:
    if hasattr(prompt, "to_ints"):
        return len(prompt.to_ints())
    if hasattr(prompt, "length"):
        return int(prompt.length)
    raise TypeError(f"Cannot get token length from {type(prompt)!r}")


def estimate_sample_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    pricing_usd_per_mtok: dict[str, float],
) -> dict[str, Any]:
    prefill_cost = prompt_tokens / 1_000_000 * pricing_usd_per_mtok["prefill"]
    sample_cost = completion_tokens / 1_000_000 * pricing_usd_per_mtok["sample"]
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prefill_usd_per_mtok": pricing_usd_per_mtok["prefill"],
        "sample_usd_per_mtok": pricing_usd_per_mtok["sample"],
        "prefill_cost_usd": prefill_cost,
        "sample_cost_usd": sample_cost,
        "estimated_cost_usd": prefill_cost + sample_cost,
    }


async def call_tinker(
    *,
    sampling_client: Any,
    renderer: Any,
    messages: list[dict[str, str]],
    repetitions: int,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> tuple[list[Any], int, float, str, str]:
    prompt = renderer.build_generation_prompt(messages)
    prompt_tokens = token_count(prompt)
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )

    started_at = utc_now_iso()
    t0 = time.time()
    result = await asyncio.wait_for(
        sampling_client.sample_async(
            prompt=prompt,
            sampling_params=params,
            num_samples=repetitions,
        ),
        timeout=timeout_s,
    )
    latency_s = time.time() - t0
    completed_at = utc_now_iso()
    return list(result.sequences), prompt_tokens, latency_s, started_at, completed_at


def make_error_result(
    *,
    error: str,
    attempts: int,
    sample_index: int,
    repetitions: int,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    return {
        "answer": "",
        "error": error,
        "attempts": attempts,
        "parse_success": False,
        "finish_reason": "error",
        "latency_s": None,
        "started_at": None,
        "completed_at": utc_now_iso(),
        "token_usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "num_samples_requested": repetitions,
            "sample_index": sample_index,
        },
        "price_estimate": None,
        "sampling_details": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_samples": repetitions,
            "sample_index": sample_index,
            "request_timeout_s": timeout_s,
        },
    }


async def sample_with_retries(
    *,
    sampling_client: Any,
    renderer: Any,
    messages: list[dict[str, str]],
    repetitions: int,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    retries: int,
    pricing_usd_per_mtok: dict[str, float],
) -> list[dict[str, Any]]:
    last_error = "unknown error"

    for attempt in range(1, retries + 1):
        try:
            seqs, prompt_tokens, latency_s, started_at, completed_at = await call_tinker(
                sampling_client=sampling_client,
                renderer=renderer,
                messages=messages,
                repetitions=repetitions,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )

            results: list[dict[str, Any]] = []
            for sample_index, seq in enumerate(seqs[:repetitions]):
                completion_tokens = len(seq.tokens)
                parsed_message, parse_success = renderer.parse_response(seq.tokens)
                results.append(
                    {
                        "answer": message_content_to_text(parsed_message),
                        "error": None,
                        "attempts": attempt,
                        "parse_success": bool(parse_success),
                        "finish_reason": getattr(seq, "stop_reason", None),
                        "latency_s": latency_s,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "token_usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "num_samples_requested": repetitions,
                            "sample_index": sample_index,
                        },
                        "price_estimate": estimate_sample_cost(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            pricing_usd_per_mtok=pricing_usd_per_mtok,
                        ),
                        "sampling_details": {
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "num_samples": repetitions,
                            "sample_index": sample_index,
                            "request_timeout_s": timeout_s,
                        },
                    }
                )

            while len(results) < repetitions:
                sample_index = len(results)
                results.append(
                    make_error_result(
                        error=f"Tinker returned {len(seqs)} sequences for num_samples={repetitions}",
                        attempts=attempt,
                        sample_index=sample_index,
                        repetitions=repetitions,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout_s=timeout_s,
                    )
                )

            return results

        except Exception as exc:
            last_error = repr(exc)
            if attempt < retries:
                await asyncio.sleep(min(60, 2 ** (attempt - 1)))

    return [
        make_error_result(
            error=last_error,
            attempts=retries,
            sample_index=sample_index,
            repetitions=repetitions,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        for sample_index in range(repetitions)
    ]


def build_output_row(
    *,
    ex: dict[str, Any],
    row_index: int,
    repetition_index: int,
    generator: str,
    base_model: str,
    renderer_name: str,
    run_info: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": ex["session_id"],
        "generator": generator,
        "output": [result["answer"]],
        "row_index": row_index,
        "repetition_index": repetition_index,
        "provider": "tinker",
        "provider_mode": "base_model",
        "provider_model": base_model,
        "recommended_renderer": renderer_name,
        "finish_reason": result["finish_reason"],
        "error": result["error"],
        "attempts": result["attempts"],
        "parse_success": result["parse_success"],
        "latency_s": result["latency_s"],
        "started_at": result["started_at"],
        "completed_at": result["completed_at"],
        "token_usage": result["token_usage"],
        "price_estimate": result["price_estimate"],
        "sampling_details": result["sampling_details"],
        "run_info": run_info,
    }


def load_existing_outputs(
    *,
    responses_dir: Path,
    repetitions: int,
    resolved_limit: int,
    overwrite: bool,
) -> tuple[list[list[dict[str, Any]]], int]:
    paths = [responses_dir / f"rep_{rep:02d}.json" for rep in range(repetitions)]

    if overwrite:
        for path in paths:
            path.unlink(missing_ok=True)

    rows_by_rep = [load_json_list(path) for path in paths]
    for path, rows in zip(paths, rows_by_rep):
        if len(rows) > resolved_limit:
            raise ValueError(f"{path} has {len(rows)} rows, but limit is {resolved_limit}")

    start_index = min((len(rows) for rows in rows_by_rep), default=0)
    if any(len(rows) != start_index for rows in rows_by_rep):
        print(f"Rep files have unequal lengths; truncating to {start_index} rows")
        for path, rows in zip(paths, rows_by_rep):
            del rows[start_index:]
            atomic_write_json(path, rows)

    return rows_by_rep, start_index


async def generate(
    *,
    bench: Any,
    sampling_client: Any,
    renderer: Any,
    renderer_name: str,
    generator: str,
    base_model: str,
    pricing_usd_per_mtok: dict[str, float],
    responses_dir: Path,
    repetitions: int,
    requested_limit: int | None,
    resolved_limit: int,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    retries: int,
    timeout_s: float,
    overwrite: bool,
    verbose: bool,
) -> None:
    responses_dir.mkdir(parents=True, exist_ok=True)
    rep_paths = [responses_dir / f"rep_{rep:02d}.json" for rep in range(repetitions)]
    rows_by_rep, start_index = load_existing_outputs(
        responses_dir=responses_dir,
        repetitions=repetitions,
        resolved_limit=resolved_limit,
        overwrite=overwrite,
    )

    run_info = {
        "script": Path(__file__).name,
        "created_at": utc_now_iso(),
        "provider": "tinker",
        "provider_mode": "base_model",
        "generator": generator,
        "base_model": base_model,
        "recommended_renderer": renderer_name,
        "dataset": constants.WILDBENCH_HF_DATASET,
        "dataset_config": constants.WILDBENCH_HF_CONFIG,
        "dataset_split": constants.WILDBENCH_HF_SPLIT,
        "dataset_size": len(bench),
        "requested_limit": requested_limit,
        "resolved_limit": resolved_limit,
        "repetitions": repetitions,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "pricing_usd_per_mtok": pricing_usd_per_mtok,
        "price_estimate_formula": "prompt_tokens / 1e6 * prefill + completion_tokens / 1e6 * sample",
        "concurrency": concurrency,
        "retries": retries,
        "request_timeout_s": timeout_s,
        "uses_num_samples_for_repetitions": True,
        "num_samples_per_request": repetitions,
    }

    semaphore = asyncio.Semaphore(concurrency)

    async def process_index(i: int) -> tuple[int, list[dict[str, Any]]]:
        ex = bench[i]
        if verbose:
            print(f"row={i}/{resolved_limit - 1} session_id={ex['session_id']} num_samples={repetitions}")

        async with semaphore:
            sample_results = await sample_with_retries(
                sampling_client=sampling_client,
                renderer=renderer,
                messages=normalize_messages(ex["conversation_input"]),
                repetitions=repetitions,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                retries=retries,
                pricing_usd_per_mtok=pricing_usd_per_mtok,
            )

        return i, [
            build_output_row(
                ex=ex,
                row_index=i,
                repetition_index=rep,
                generator=generator,
                base_model=base_model,
                renderer_name=renderer_name,
                run_info=run_info,
                result=sample_results[rep],
            )
            for rep in range(repetitions)
        ]

    pending: dict[int, list[dict[str, Any]]] = {}
    next_to_write = start_index
    tasks = [asyncio.create_task(process_index(i)) for i in range(start_index, resolved_limit)]

    with tqdm(total=resolved_limit, initial=start_index, desc=f"{generator} num_samples={repetitions}") as pbar:
        for task in asyncio.as_completed(tasks):
            i, rows_for_reps = await task
            pending[i] = rows_for_reps
            pbar.update(1)

            while next_to_write in pending:
                for rep, row in enumerate(pending.pop(next_to_write)):
                    rows_by_rep[rep].append(row)
                next_to_write += 1

            for rep, path in enumerate(rep_paths):
                atomic_write_json(path, rows_by_rep[rep])

    total_errors = sum(1 for rows in rows_by_rep for row in rows if row.get("error"))
    print(f"Wrote {repetitions} repetition files under {responses_dir}")
    print(f"Rows per repetition: {len(rows_by_rep[0]) if rows_by_rep else 0}")
    print(f"Errored rows across all reps: {total_errors}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True)
    parser.add_argument("--generators-file", default=constants.DEFAULT_GENERATORS_REGISTRY)
    parser.add_argument("--responses-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=constants.DEFAULT_REPETITIONS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=constants.DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=constants.DEFAULT_MAX_TOKENS)
    parser.add_argument("--concurrency", type=int, default=constants.DEFAULT_CONCURRENCY)
    parser.add_argument("--retries", type=int, default=constants.DEFAULT_RETRIES_GENERATE)
    parser.add_argument("--request-timeout-s", type=float, default=constants.DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.repetitions < 1:
        raise ValueError("--repetitions must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    generator_cfg = load_generator_config(Path(args.generators_file), args.generator)
    base_model = generator_cfg["base_model"]
    pricing_usd_per_mtok = load_pricing(generator_cfg)

    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(base_model=base_model)
    tokenizer = sampling_client.get_tokenizer()
    renderer_name = get_recommended_renderer_name(base_model)
    renderer = get_renderer(renderer_name, tokenizer)

    bench = load_dataset(
        constants.WILDBENCH_HF_DATASET,
        constants.WILDBENCH_HF_CONFIG,
        split=constants.WILDBENCH_HF_SPLIT,
        token=os.environ.get("HF_TOKEN"),
    )
    resolved_limit = len(bench) if args.limit is None else min(args.limit, len(bench))

    responses_dir = Path(args.responses_dir) / args.generator
    manifest = {
        "script": Path(__file__).name,
        "created_at": utc_now_iso(),
        "generator": args.generator,
        "base_model": base_model,
        "recommended_renderer": renderer_name,
        "dataset": constants.WILDBENCH_HF_DATASET,
        "dataset_config": constants.WILDBENCH_HF_CONFIG,
        "dataset_split": constants.WILDBENCH_HF_SPLIT,
        "dataset_size": len(bench),
        "requested_limit": args.limit,
        "resolved_limit": resolved_limit,
        "repetitions": args.repetitions,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "pricing_usd_per_mtok": pricing_usd_per_mtok,
        "price_estimate_formula": "prompt_tokens / 1e6 * prefill + completion_tokens / 1e6 * sample",
        "concurrency": args.concurrency,
        "retries": args.retries,
        "request_timeout_s": args.request_timeout_s,
        "uses_num_samples_for_repetitions": True,
        "num_samples_per_request": args.repetitions,
        "output_files": [str(responses_dir / f"rep_{rep:02d}.json") for rep in range(args.repetitions)],
    }
    atomic_write_json(responses_dir / "manifest.json", manifest)

    await generate(
        bench=bench,
        sampling_client=sampling_client,
        renderer=renderer,
        renderer_name=renderer_name,
        generator=args.generator,
        base_model=base_model,
        pricing_usd_per_mtok=pricing_usd_per_mtok,
        responses_dir=responses_dir,
        repetitions=args.repetitions,
        requested_limit=args.limit,
        resolved_limit=resolved_limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        retries=args.retries,
        timeout_s=args.request_timeout_s,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    asyncio.run(main())
