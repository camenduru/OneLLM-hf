---
title: OneLLM
emoji: 🚀
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
---

# OneLLM: One Framework to Align All Modalities with Language

[[Project Page](https://onellm.csuhan.com)] [[Paper](#)] [[Web Demo](https://huggingface.co/spaces/csuhan/OneLLM)]

Authors: [Jiaming Han](), [Kaixiong Gong](), [Yiyuan Zhang](), [Jiaqi Wang](), [Kaipeng Zhang](), [Dahua Lin](), [Yu Qiao](), [Peng Gao](), [Xiangyu Yue]().

## News

- **2023.12.01** Release model weights and inference code.

## Contents

- [Install](#install)
- [Models](#models)
- [Demo](#demo)

<!-- - [Evaluation](#evaluation) -->

<!-- - [Training](#training) -->

### TODO

- [ ] Data
- [ ] Evaluation
- [ ] Training

### Install

1. Clone the repo into a local folder.

```bash
git clone https://github.com/csuhan/OneLLM

cd OneLLM
```

2. Install packages.

```bash
conda create -n onellm python=3.9 -y
conda activate onellm

pip install -r requirements.txt

# install pointnet
cd lib/pointnet2
python setup.py install
```

3. Install Apex. (Optional)

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

### Models

We provide a preview model at: [csuhan/OneLLM-7B](https://huggingface.co/csuhan/OneLLM-7B).

### Demo

**Huggingface Demo:** [csuhan/OneLLM](https://huggingface.co/spaces/csuhan/OneLLM).

**Local Demo:** Assume you have downloaded the weights to ${WEIGHTS_DIR}. Then run the following command to start a gradio demo locally.

```bash
python demos/multi_turn_mm.py --gpu_ids 0 --tokenizer_path config/llama2/tokenizer.model --llama_config config/llama2/7B.json --pretrained_path ${WEIGHTS_DIR}/consolidated.00-of-01.pth
```

<!-- ### Evaluation -->

<!-- ### Training -->

## Citation

```
@article{han2023onellm,
  title={OneLLM: One Framework to Align All Modalities with Language},
  author={Han, Jiaming and Gong, Kaixiong and Zhang, Yiyuan and Wang, Jiaqi and Zhang, Kaipeng and Lin, Dahua and Qiao, Yu and Gao, Peng and Yue, Xiangyu},
  journal={arXiv preprint arXiv:xxxx},
  year={2023}
}
```

## Acknowledgement

[LLaMA](https://github.com/facebookresearch/llama), [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter), [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory), [Meta-Transformer](https://github.com/invictus717/MetaTransformer), [ChatBridge](https://github.com/joez17/ChatBridge)
