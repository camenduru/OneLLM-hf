#!/usr/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=1

python /home/pgao/leimeng/PandaLM/pandalm/utils/pandalm_inference.py \
    --model_name=WeOpenML/PandaLM-7B-v1 \
    --input_path llama_adapter_vs_alpaca_lora.json \
    --output_path llama_adapter_vs_alpaca_lora_eval.json


# llama_adapter vs alpaca {'win': 47, 'lose': 45, 'tie': 68}
# llama_adapter vs alpaca lora {'win': 54, 'lose': 43, 'tie': 63}
