# Market for LMs


1. Collect responses
```
python scripts/01_run_wildbench_tinker.py \
  --generators-file configs/tinker_generators.json \
  --responses-dir results/responses/passk10 \
  --repetitions 10 \
  --temperature 1 \
  --max-tokens 4096 \
  --concurrency 16 \
  --request-timeout-s 300 \
  --retries 2
```


2. Create eval batch submissions
```
python scripts/02_run_wildbench_eval.py \
  --responses-dir results/responses/passk10 \
  --evals-dir results/evals/passk10 \
  --wildbench-dir external/WildBench \
  --generators-file configs/tinker_generators.json \
  --judge-model gpt-5.4-nano \
  --overwrite
```

3. Patch batch submission (gpt4 -> gpt5 schema)
```
python scripts/03_patch_gpt5_batch_schema.py \
  --evals-dir results/evals/passk10 \
  --judge-model gpt-5.4-nano
```

4. Submit batches until completed (rate limiting)
```
while true; do
  echo "=== $(date) ==="

  python scripts/04_resubmit_wildbench_failed_batches.py \
    --evals-dir results/evals/passk10 \
    --wildbench-dir external/WildBench \
    --judge-model gpt-5.4-nano \
    --max-submit 3

  echo "=== sleeping 10 minutes ==="
  sleep 600
done
```

5. Extract completed evals
```
python scripts/05_extract_wildbench_batch_results.py \
  --evals-dir results/evals/passk10 \
  --generators-file configs/tinker_generators.json \
  --judge-model gpt-5.4-nano
```

