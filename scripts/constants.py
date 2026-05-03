"""Shared defaults for WildBench / Tinker pipeline scripts."""

from __future__ import annotations

# --- Paths ---
DEFAULT_RESPONSES_DIR = "results/responses/passk10"
DEFAULT_EVALS_DIR = "results/evals/passk10"
DEFAULT_GENERATORS_REGISTRY = "configs/tinker_generators.json"
DEFAULT_GENERATOR_DRIVER_SCRIPT = "scripts/generate_wildbench_tinker_base_repetitions.py"
DEFAULT_WILDBENCH_REPO_DIR = "external/WildBench"
DEFAULT_EVAL_TEMPLATE_RELATIVE = "evaluation/eval_template.score.v2.md"

# --- Judge / eval ---
DEFAULT_JUDGE_MODEL = "gpt-5.4-nano"
DEFAULT_FAILURE_RAW_SCORE = 1.0

# --- Default generator keys (eval / batch extract when not using full registry) ---
DEFAULT_GENERATORS: list[str] = [
    "tinker__llama_3_2_1b",
    "tinker__qwen3_4b_instruct_2507",
    "tinker__qwen3_30b_a3b_instruct_2507",
    "tinker__gpt_oss_20b",
    "tinker__nemotron_3_nano_30b_a3b_bf16",
    "tinker__llama_3_1_8b_instruct",
    "tinker__qwen3_8b",
    "tinker__nemotron_3_super_120b_a12b_bf16",
    "tinker__deepseek_v3_1",
    "tinker__kimi_k2_5",
]

# --- OpenAI Batch API ---
OPENAI_BATCHES_LIST_PAGE_LIMIT_MAX = 100
OPENAI_BATCH_LIST_PAGE_DEFAULT = 100

# States for extract script (includes cancelling as terminal).
OPENAI_BATCH_TERMINAL_FAILURE_STATUSES = frozenset({"failed", "expired", "cancelled", "cancelling"})
OPENAI_BATCH_ACTIVE_STATUSES = frozenset({"validating", "in_progress", "finalizing", "cancelling"})
OPENAI_BATCH_RESUBMIT_TERMINAL_BAD = frozenset({"failed", "expired", "cancelled"})

# --- GPT-5 batch JSONL body keys removed by 03_patch_gpt5_batch_schema ---
GPT5_BATCH_SUBMIT_BODY_DROP_KEYS = frozenset(
    {
        "stream",
        "stream_options",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "parallel_tool_calls",
    }
)

# --- Tinker generation (generate + 01 runner CLI defaults) ---
DEFAULT_REPETITIONS = 10
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONCURRENCY = 4
DEFAULT_REQUEST_TIMEOUT_S = 300.0
DEFAULT_RETRIES_GENERATE = 2
DEFAULT_RETRIES_TINKER_RUNNER = 5

DEFAULT_MAX_SUBMIT_BATCHES = 5

# --- Hugging Face WildBench dataset (generate) ---
WILDBENCH_HF_DATASET = "allenai/WildBench"
WILDBENCH_HF_CONFIG = "v2"
WILDBENCH_HF_SPLIT = "test"
