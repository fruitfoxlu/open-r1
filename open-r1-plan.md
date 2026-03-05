# Open-R1 Execution Plan (This Machine)

## Goal
Execute and report two experiments end-to-end on this 8xH100 server:
1. `SFT` with smaller student model (`Qwen/Qwen3-0.6B-Base`)
2. `GRPO code reward` (`config_demo_code.yaml`) with **local Docker provider**

## Shared Preflight (run once)
1. Environment
```bash
eval "$(conda shell.bash hook)"
conda activate openr1-conda
cd /home/rlu/Code/open-r1
export PYTHONPATH=src
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_DATASETS_CACHE=/tmp/hf_cache/datasets
export TRANSFORMERS_CACHE=/tmp/hf_cache/transformers
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```
2. Auth
```bash
hf auth login
wandb login
```
3. Docker check (required for local code provider)
```bash
docker run --rm python:3.11-slim python3 -c "print('ok')"
```

## Common Evaluation Protocol (before/after)
Use identical LightEval settings for baseline and trained model.

```bash
MODEL="<model_id_or_local_path>"
OUT="data/evals/${MODEL//\//_}"
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

lighteval vllm $MODEL_ARGS "lighteval|aime24|0|0" --use-chat-template --output-dir "$OUT"
lighteval vllm $MODEL_ARGS "lighteval|math_500|0|0" --use-chat-template --output-dir "$OUT"
lighteval vllm $MODEL_ARGS "lighteval|gpqa:diamond|0|0" --use-chat-template --output-dir "$OUT"
lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" --use-chat-template --output-dir "$OUT"
```

Record before/after metrics in one markdown table.

---

## 1) SFT (override smaller model, 0.6B)
### Train
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  src/open_r1/sft.py \
  --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml \
  --model_name_or_path Qwen/Qwen3-0.6B-Base \
  --hub_model_id <HF_NAMESPACE>/OpenR1-Distill-0.6B \
  --output_dir data/OpenR1-Distill-0.6B \
  --push_to_hub false \
  --report_to wandb
```

### Evaluate
- Baseline: `Qwen/Qwen3-0.6B-Base`
- After: `data/OpenR1-Distill-0.6B`
- Run common evaluation protocol.

### Success criteria (publish gate)
- Train completes without divergence/NaN.
- At least **3/4** benchmark deltas are non-negative.
- Average delta across `{aime24, math_500, gpqa, lcb}` is **>= +1.0**.

### Publish
```bash
REPO="<HF_NAMESPACE>/OpenR1-Distill-0.6B"
hf repo create "$REPO" --type model --private=false || true
huggingface-cli upload "$REPO" data/OpenR1-Distill-0.6B . --repo-type model
```
Add model card section: training config, hardware, before/after table, pass/fail vs gate.

---

## 2) GRPO code reward (`config_demo_code.yaml`) with local provider
### Local provider precheck
```bash
python - <<'PY'
from open_r1.rewards import code_reward
completions=[[{"role":"assistant","content":"```python\nprint('ok')\n```"}]]
verification_info=[{"language":"python","test_cases":[{"input":"","output":"ok"}]}]
print(code_reward(completions, provider_type='local', verification_info=verification_info, num_parallel=1))
PY
```
Expected output includes `[1.0]`.

### Train
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes 8 \
  src/open_r1/grpo.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code.yaml \
  --code_provider local \
  --report_to wandb \
  --push_to_hub false \
  --output_dir data/Qwen2.5-1.5B-Open-R1-Code-GRPO-local
```

### Evaluate
- Baseline: `Qwen/Qwen2.5-1.5B-Instruct`
- After: `data/Qwen2.5-1.5B-Open-R1-Code-GRPO-local`
- Run common evaluation protocol.
- Primary KPI: `lcb` (and optionally `lcb_v4` if needed).

### Success criteria (publish gate)
- Train completes without repeated provider failures.
- `lcb` absolute gain **>= +1.5** over baseline.
- No catastrophic regression: `math_500` drop not worse than **-2.0**.

### Publish
```bash
REPO="<HF_NAMESPACE>/Qwen2.5-1.5B-Open-R1-Code-GRPO-local"
hf repo create "$REPO" --type model --private=false || true
huggingface-cli upload "$REPO" data/Qwen2.5-1.5B-Open-R1-Code-GRPO-local . --repo-type model
```
Add model card section: local-provider hardening profile, training config, before/after table, success gate outcome.

---

## Result Artifact Requirements
For each experiment, save:
- final command line used
- git commit hash
- wall-clock time
- before/after metrics table
- decision: `PASS` / `FAIL` against gate

Suggested file: `reports/<exp_name>_result.md`

## Questions Before Execution
1. What Hugging Face namespace should we publish to (`<HF_NAMESPACE>`) ?
2. Public or private model repos?
3. Keep the success gates above, or do you want stricter/looser thresholds?
