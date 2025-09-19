import os
import sys
import torch
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
import umap as um

def square_crop_image(image: Image.Image, xslide=0, yslide=0):
    side_length = min(image.width, image.height)
    wm, hm, rd = image.width // 2, image.height // 2, side_length // 2
    return image.crop((wm - rd - xslide, hm - rd - yslide, wm + rd - xslide, hm + rd - yslide))

model_path = '/home/tumai/models/Qwen--Qwen2.5-VL-7B-Instruct'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto"
)
patch_size, spatial_merge = 14, 2 # model hyperparams
processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
model.eval()

def qwen_encode_image(image: Image.Image):
    image_inputs = processor.image_processor( #type:ignore
        images=[image], return_tensors="pt"
    )
    pixel_values = image_inputs["pixel_values"].to(model.device).to(torch.bfloat16)
    image_grid_thw = image_inputs["image_grid_thw"].to(model.device)
    with torch.no_grad():
        return model.visual(pixel_values, image_grid_thw)

def patches_to_2d(patch_features: torch.Tensor, src_image: Image.Image):
    effective_patch_size = patch_size * spatial_merge
    patches_width, patches_height = src_image.width // effective_patch_size, src_image.height // effective_patch_size
    return patch_features.reshape(patches_height, patches_width, -1)

def patch_cds(src_image: Image.Image):
    effective_patch_size = patch_size * spatial_merge
    patches_width, patches_height = src_image.width // effective_patch_size, src_image.height // effective_patch_size
    patch_coords_list = []
    for i in range(patches_height):
        for j in range(patches_width):
            y1 = i * effective_patch_size
            y2 = min((i + 1) * effective_patch_size, src_image.height)
            x1 = j * effective_patch_size
            x2 = min((j + 1) * effective_patch_size, src_image.width)
            patch_coords_list.append((y1, y2, x1, x2))
    cds: torch.Tensor = torch.as_tensor(patch_coords_list)
    cds: torch.Tensor = cds.reshape(patches_height, patches_width, 2, 2)
    return cds

# This function takes in an RGB image  and a prompt
def ask_qwen_about_image(image: Image.Image, prompt: str):
    messages = [{"role": "user", "content":
            [{"type": "image","image": None},
             {"type": "text", "text": prompt},]
    }]
    text = processor.apply_chat_template( #type:ignore
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# This function takes in the **patch features** of an image and a prompt
def ask_qwen_about_image_features(image_features: torch.Tensor, prompt: str):
    messages = [{"role": "user", "content":
            [{"type": "image","image": None},
             {"type": "text", "text": prompt},]
    }]
    text = processor.apply_chat_template( #type:ignore
        messages, tokenize=False, add_generation_prompt=True
    )
    def closest_factor_pair(n) -> tuple[int, int]:
        root = int(math.isqrt(n))
        for a in range(root, 0, -1):
            if n % a == 0:
                return a, n // a
        raise Exception("the given feature patches don't correspond to a nice rectangular size in pixels")
    effective_patch_size = patch_size * spatial_merge
    pw_dummy, ph_dummy = closest_factor_pair(image_features.shape[0])
    inputs = processor(
        text=[text],
        images=[Image.new("RGB", (effective_patch_size * pw_dummy, effective_patch_size * ph_dummy), color="red")],
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=128, custom_patch_features=image_features)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def crop_patch_features(patch_feat: torch.Tensor, cw, ch, cx1, cx2, cy1, cy2):
    return patch_feat.reshape(ch, cw, -1)[cy1:cy2+1, cx1:cx2+1].flatten(end_dim=1)