import sys
import os

import argparse
import multiprocessing as mp
import numpy as np
from typing import List, Optional

import torch
import torch.distributed as dist

from fairscale.nn.model_parallel import initialize as fs_init

import gradio as gr
from util.misc import setup_for_distributed
from util.misc import default_tensor_type
from model.meta import MetaModel
from data.conversation_lib import conv_templates, SeparatorStyle
from PIL import Image
import torchvision.transforms as transforms
from data.fintune_dataset import make_audio_features
from data import video_utils 
from dataclasses import dataclass
from huggingface_hub import hf_hub_download

T_random_resized_crop = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=3,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


def load_audio(audio_path):
    fbank = make_audio_features(audio_path, mel_bins=128)
    fbank = fbank.transpose(0, 1)[None] #[1, 128, 1024]
    return fbank
    
def load_video(video_path):
    video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
    return video_feats[:, :, 0]


def model_worker(
    rank: int, args: argparse.Namespace, barrier: mp.Barrier,
    request_queue: mp.Queue, response_queue: Optional[mp.Queue] = None,
) -> None:
    """
    The worker function that manipulates the GPU to run the inference.
    Exact n_gpu workers are started, with each one operating on a separate GPU.

    Args:
        rank (int): Distributed rank of the worker.
        args (argparse.Namespace): All command line arguments.
        barrier (multiprocessing.Barrier): A barrier used to delay the start
            of Web UI to be after the start of the model.
    """

    world_size = len(args.gpu_ids)
    gpu_id = args.gpu_ids[rank]
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
    )
    print(f"| distributed init on worker {rank}/{world_size}. "
          f"using gpu: {gpu_id}")
    fs_init.initialize_model_parallel(world_size)
    torch.cuda.set_device(gpu_id)

    torch.manual_seed(1)
    np.random.seed(1)

    # set the print behavior.
    setup_for_distributed(rank == 0)

    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }[args.dtype]
    with default_tensor_type(dtype=target_dtype, device="cuda"):
        model = MetaModel(args.llama_type, args.llama_config, tokenizer_path=args.tokenizer_path)
    for ckpt_id in range(args.num_ckpts):
        ckpt_path = hf_hub_download(repo_id=args.pretrained_path, filename=args.ckpt_format.format(str(ckpt_id)))
        print(f"Loading pretrained weights {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
    # print("load result:\n", msg)
    model.cuda()
    model.eval()
    print(f"Model = {str(model)}")

    barrier.wait()

    while True:
        img_path, audio_path, video_path, chatbot, max_gen_len, temperature, top_p, modality = request_queue.get()
        if 'image' in modality and img_path is not None:
            image = Image.open(img_path).convert('RGB')
            inputs = T_random_resized_crop(image)
        elif 'video' in modality and video_path is not None:
            inputs = load_video(video_path)
        elif 'audio' in modality and audio_path is not None:
            inputs = load_audio(audio_path)
        else:
            inputs = None
        
        if inputs is not None:
            inputs = inputs[None].cuda().to(target_dtype)
    
        conv = conv_templates["v1"].copy()
        for user, bot in chatbot:
            conv.append_message(conv.roles[0], user)
            conv.append_message(conv.roles[1], bot)

        with torch.cuda.amp.autocast(dtype=target_dtype):
            print(conv.get_prompt())
            for stream_response in model.stream_generate(
                conv.get_prompt(), inputs,
                max_gen_len=max_gen_len, temperature=temperature, top_p=top_p,
                modal = modality
            ):
                conv_sep = (
                    conv.sep
                    if conv.sep_style == SeparatorStyle.SINGLE
                    else conv.sep2
                )
                end_pos = stream_response["text"].find(conv_sep)
                if end_pos != -1:
                    stream_response["text"] = (
                        stream_response['text'][:end_pos].rstrip() + "\n"
                    )
                    stream_response["end_of_content"] = True

                # keep a few characters if not end_of_content to avoid sending
                # part of conv_sep before all of it is generated.
                if not stream_response["end_of_content"]:
                    if len(stream_response["text"]) < len(conv_sep):
                        continue
                    stream_response["text"] = (
                        stream_response["text"][:-len(conv_sep)]
                    )

                if response_queue is not None:
                    response_queue.put(stream_response)

                if stream_response["end_of_content"]:
                    break


def gradio_worker(
    request_queues: List[mp.Queue], response_queue: mp.Queue,
    args: argparse.Namespace, barrier: mp.Barrier,
) -> None:
    """
    The gradio worker is responsible for displaying the WebUI and relay the
    requests to model workers. It should be launched only once.

    Args:
        request_queues (List[mp.Queue]): A list of request queues (one for
            each model worker).
        args (argparse.Namespace): All command line arguments.
        barrier (multiprocessing.Barrier): A barrier used to delay the start
            of Web UI to be after the start of the model.
    """

    def show_user_input(msg, chatbot):
        return "", chatbot + [[msg, None]]

    def stream_model_output(img_path, audio_path, video_path, chatbot, max_gen_len, gen_t, top_p, modality):
        for queue in request_queues:
            queue.put((img_path, audio_path, video_path, chatbot, max_gen_len, gen_t, top_p, modality))
        while True:
            content_piece = response_queue.get()
            chatbot[-1][1] = content_piece["text"]
            yield chatbot
            if content_piece["end_of_content"]:
                break

    def undo(chatbot):
        if len(chatbot) > 0:
            chatbot = chatbot[:-1]
        return chatbot

    def clear():
        chatbot = []
        msg = ""
        return chatbot, msg
    
    def change_modality(tab):
        modality = 'image'
        label_modal_dict = {
            'Image': 'image',
            'Video': 'video',
            'Audio': 'audio',
            'Point Cloud': 'point',
            'IMU': 'imu',
            'fMRI': 'fmri',
            'Depth Map': 'rgbd',
            'Normal Map': 'rgbn'
        }
        print(tab.label)
        if tab.label in label_modal_dict:
            modality = label_modal_dict[tab.label]
        return modality

    CSS ="""
    .contain { display: flex; flex-direction: column; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; overflow: auto;}
    """

    with gr.Blocks(css=CSS, theme=gr.themes.Base()) as demo:
        gr.Markdown("## OneLLM: One Framework to Align All Modalities with Language")
        with gr.Row(equal_height=True):
            # with gr.Column(scale=1):
            #     img_path = gr.Image(label='Image Input', type='filepath')
            #     video_path = gr.Video(label='Video Input')
            #     audio_path = gr.Audio(label='Audio Input', type='filepath', sources=['upload'])
            # modality = gr.Radio(choices=['image', 'audio', 'video'], value='image', interactive=True, label='Input Modalities', visible=False)
            modality = gr.Textbox(value='image', visible=False)
            with gr.Column(scale=1):
                with gr.Tab('Image') as img_tab:
                    img_path = gr.Image(label='Image Input', type='filepath')
                with gr.Tab('Video') as video_tab:
                    video_path = gr.Video(label='Video Input')
                with gr.Tab('Audio') as audio_tab:
                    audio_path = gr.Audio(label='Audio Input', type='filepath', sources=['upload'])

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(elem_id="chatbot")
                msg = gr.Textbox()

        with gr.Row():
            submit_button = gr.Button("Submit", variant="primary")
            undo_button = gr.Button("Undo")
            clear_button = gr.ClearButton([chatbot, msg, img_path, audio_path, video_path, modality])
        with gr.Row():
            max_gen_len = gr.Slider(
                minimum=1, maximum=args.model_max_seq_len // 2,
                value=args.model_max_seq_len // 2, interactive=True,
                label="Single-turn max response length",
            )
            gen_t = gr.Slider(
                minimum=0, maximum=1, value=0.1, interactive=True,
                label="Temperature",
            )
            top_p = gr.Slider(
                minimum=0, maximum=1, value=0.75, interactive=True,
                label="Top-p",
            )
        
        img_tab.select(change_modality, [img_tab], [modality])
        video_tab.select(change_modality, [video_tab], [modality])
        audio_tab.select(change_modality, [audio_tab], [modality])

        msg.submit(
            show_user_input, [msg, chatbot], [msg, chatbot],
        ).then(
            stream_model_output, [img_path, audio_path, video_path, chatbot, max_gen_len, gen_t, top_p, modality], chatbot,
        )
        submit_button.click(
            show_user_input, [msg, chatbot], [msg, chatbot],
        ).then(
            stream_model_output, [img_path, audio_path, video_path, chatbot, max_gen_len, gen_t, top_p, modality], chatbot,
        )
        undo_button.click(undo, chatbot, chatbot)
        # img_path.change(clear, [], [chatbot, msg])
    barrier.wait()
    demo.queue(api_open=True).launch(share=True, max_threads=1)


@dataclass
class DemoConfig:
    gpu_ids = [0]
    tokenizer_path = "config/llama2/tokenizer.model"
    llama_type = "onellm"
    llama_config = "config/llama2/7B.json"
    model_max_seq_len = 2048
    # pretrained_path = "weights/7B_2048/consolidated.00-of-01.pth"
    # pretrained_path = hf_hub_download(repo_id="csuhan/OneLLM-7B", filename="consolidated.00-of-01.pth")
    pretrained_path = "csuhan/OneLLM-7B-hf"
    ckpt_format = "consolidated.00-of-01.s{}.pth"
    num_ckpts = 10
    master_port = 23861
    master_addr = "127.0.0.1"
    dtype = "fp16"

if __name__ == "__main__":
    args = DemoConfig()
    # using the default "fork" method messes up some imported libs (e.g.,
    # pandas)
    # mp.set_start_method("spawn")

    # setup the queues and start the model workers
    request_queues = []
    response_queue = mp.Queue()
    worker_processes = []
    barrier = mp.Barrier(len(args.gpu_ids) + 1)
    for rank, gpu_id in enumerate(args.gpu_ids):
        request_queue = mp.Queue()
        rank_response_queue = response_queue if rank == 0 else None
        process = mp.Process(
            target=model_worker,
            args=(rank, args, barrier, request_queue, rank_response_queue),
        )
        process.start()
        worker_processes.append(process)
        request_queues.append(request_queue)

    gradio_worker(request_queues, response_queue, args, barrier)
