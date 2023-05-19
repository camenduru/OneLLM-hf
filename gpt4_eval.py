import json
import os
import glob
import sys
import time
from pathlib import Path
from typing import Tuple

import shortuuid
# from huggingface_hub import hf_hub_download
from PIL import Image
import gradio as gr
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import LLaMA, ModelArgs, Tokenizer, Transformer, VisionModel

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def setup_model_parallel() -> Tuple[int, int]:
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MP'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '2223'
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_path: str,
    param_path: str,
    tokenizer_path: str,
    instruct_adapter_path: str,
    caption_adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    print("Loading")
    instruct_adapter_checkpoint = torch.load(
        instruct_adapter_path, map_location="cpu")
    caption_adapter_checkpoint = torch.load(
        caption_adapter_path, map_location="cpu")
    with open(param_path, "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    model_args.adapter_layer = int(
        instruct_adapter_checkpoint['adapter_query.weight'].shape[0] / model_args.adapter_len)
    model_args.cap_adapter_layer = int(
        caption_adapter_checkpoint['cap_adapter_query.weight'].shape[0] / model_args.cap_adapter_len)

    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)

    ckpt = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(ckpt, strict=False)

    vision_model = VisionModel(model_args)

    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(instruct_adapter_checkpoint, strict=False)
    model.load_state_dict(caption_adapter_checkpoint, strict=False)
    vision_model.load_state_dict(caption_adapter_checkpoint, strict=False)

    generator = LLaMA(model, tokenizer, vision_model)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def instruct_generate(
    instruct: str,
    input: str = 'none',
    max_gen_len=512,
    temperature: float = 0.1,
    top_p: float = 0.75,
):
    if input == 'none':
        prompt = PROMPT_DICT['prompt_no_input'].format_map(
            {'instruction': instruct, 'input': ''})
    else:
        prompt = PROMPT_DICT['prompt_input'].format_map(
            {'instruction': instruct, 'input': input})

    results = generator.generate(
        [prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
    )
    result = results[0].strip()
    # print(result)
    return result


ckpt_path = "/data1/llma/7B/consolidated.00.pth"
param_path = "/data1/llma/7B/params.json"
tokenizer_path = "/data1/llma/tokenizer.model"
instruct_adapter_path = "llama_adapter_len10_layer30_release.pth"
caption_adapter_path = "llama_adapter_len10_layer30_caption_vit_l.pth"
max_seq_len = 512
max_batch_size = 32


local_rank, world_size = setup_model_parallel()
if local_rank > 0:
    sys.stdout = open(os.devnull, "w")

generator = load(
    ckpt_path, param_path, tokenizer_path, instruct_adapter_path, caption_adapter_path, local_rank, world_size, max_seq_len, max_batch_size
)

answer_data = []
for line in open('question.jsonl').readlines():
    line = json.loads(line)
    question_text = line["text"]
    answer = {
        "answer_id": shortuuid.uuid(),
        "model_id": "LLaMA-Adapter",
        "question_id": line["question_id"],
        "question_text": question_text,
        "text": '',
        "metadata": {}
    }
    answer_data.append(answer)

prompts = [PROMPT_DICT['prompt_no_input'].format_map({'instruction': x['question_text']}) for x in answer_data]

results = []
result = generator.generate(prompts[:32], max_gen_len=512, temperature=0.1, top_p=0.75)
results.extend(result)
result = generator.generate(prompts[32:64], max_gen_len=512, temperature=0.1, top_p=0.75)
results.extend(result)
result = generator.generate(prompts[64:], max_gen_len=512, temperature=0.1, top_p=0.75)
results.extend(result)

for i in range(len(answer_data)):
    answer_i = answer_data[i]
    answer_i['text'] = results[i].strip()
    del answer_i['question_text']
    answer_data[i] = answer_i

with open('llama_adapter_7b.json', 'w') as f:
    f.write("\n".join([json.dumps(x) for x in answer_data]))