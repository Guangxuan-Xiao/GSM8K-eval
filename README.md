# Evaluate LLMs on GSM8k

This repository contains a minimal implementation of the evaluation code for LLMs on GSM8k.

## Requirements
PyTorch, transformers, numpy, pandas, sklearn, tqdm

## Example Usage

```bash
MODEL=meta-llama/Llama-2-7b-hf
device=0
CUDA_VISIBLE_DEVICES=$device python main.py \
    --model_name_or_path $MODEL \
    --output_dir outputs/$MODEL
```

## References

https://github.com/alibaba/FederatedScope/blob/dev/llm/federatedscope/llm/eval/eval_for_gsm8k/eval.py

https://github.com/kojima-takeshi188/zero_shot_cot