import torch
from diffusers import FluxPipeline, FluxKontextPipeline, SanaPipeline


SPEED_PIPE = None
QUALITY_PIPE = None
EDIT_PIPE = None


def get_speed_pipe():
    global SPEED_PIPE
    if SPEED_PIPE is None:
        # - Load FLUX.1-schnell
        SPEED_PIPE = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
        print("Model loaded:", "FLUX.1-schnell")
        # # - Load SANA1.5_4.8B  
        # SPEED_PIPE = SanaPipeline.from_pretrained("Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers", torch_dtype=torch.bfloat16).to("cuda")
        # SPEED_PIPE.text_encoder.to(torch.bfloat16)
        # print("Model loaded:", "SANA1.5_4.8B")
    return SPEED_PIPE


def get_quality_pipe():
    global QUALITY_PIPE
    if QUALITY_PIPE is None:
        # - Load FLUX.1-dev
        QUALITY_PIPE = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
        print("Model loaded:", "FLUX.1-dev")
    return QUALITY_PIPE


def get_edit_pipe():
    global EDIT_PIPE
    if EDIT_PIPE is None:
        # -Load FLUX.1-Kontext-dev
        EDIT_PIPE = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")
        print("Model loaded:", "FLUX.1-Kontext-dev")
    return EDIT_PIPE

