# Market for LMs


To collect queries
```
python run_wildbench_tinker.py \
  --generators-file tinker_generators.json \
  --out-dir results/passk10 \
  --repetitions 10 \
  --temperature 1 \
  --max-tokens 4096 \
  --concurrency 256 \
  --request-timeout-s 300 \
  --retries 2
```


To run evals
```

```