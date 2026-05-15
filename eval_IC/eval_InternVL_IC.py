import argparse as _argparse
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


def _exit_if_help_requested():
    if not any(arg in ("-h", "--help") for arg in _sys.argv[1:]):
        return
    parser = _argparse.ArgumentParser(description='image-captioning evaluation for InternVL.')
    parser.add_argument("--data_path", default="./data_annotation/coco_test_sub.json")
    parser.add_argument("--image_path", default="./adv_output/FOA")
    parser.add_argument("--output_dir", default="./caption_outputs")
    parser.add_argument("--model_path", default='OpenGVLab/InternVL3-8B')
    parser.add_argument("--cuda_visible_devices", default='0,1,2')
    parser.print_help()
    raise SystemExit(0)


_exit_if_help_requested()

def _parse_runtime_args():
    parser = _argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_path", default="./data_annotation/coco_test_sub.json")
    parser.add_argument("--image_path", default="./adv_output/FOA")
    parser.add_argument("--output_dir", default="./caption_outputs")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--cuda_visible_devices", default="")
    args, _ = parser.parse_known_args()
    return args

import math
import torch
import torchvision.transforms as T
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

import os
import sys
import re
import datetime
import json
from tqdm import tqdm

from dataset import paired_dataset2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
args = _parse_runtime_args()
if not args.model_path:
    args.model_path = "OpenGVLab/InternVL3-8B"
if not args.cuda_visible_devices:
    args.cuda_visible_devices = "0,1,2"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
device = 'cuda'

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

path = args.model_path
device_map = split_model(args.model_path)
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    # load_in_8bit = True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


s_test_transform = transforms.Compose([
    transforms.Resize(384, interpolation=Image.BICUBIC),
    transforms.ToTensor(),       
])
# load data
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './datasets_own/MSCOCO')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/AttackVLM')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/AnyAttack')
test_dataset = paired_dataset2(args.data_path, s_test_transform, args.image_path)
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/TYAFCQformer-ITC-0.362.5-11-17Layers')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/noise_base_test')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/noise_TYAFC')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/noise_TYAFC_test')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/MIA')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/FS')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/87.5')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/CA')
test_loader = DataLoader(test_dataset, batch_size=1,
                        num_workers=4, collate_fn=test_dataset.collate_fn)
generated_results = []
reference_dict = {}

# 创建带时间戳的目录
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(args.output_dir, timestamp)
os.makedirs(save_dir, exist_ok=True)

# 定义文件路径
adv_json_path = os.path.join(save_dir, "adv.json")
clean_json_path = os.path.join(save_dir, "clean.json")

# 处理数据
for batch in tqdm(test_loader, desc="Evaluating"):
    images, ref_captions_list, image_ids, _, image_paths = batch
    images = images.to(dtype=torch.float16).to(model.device)
    
    for i in range(images.size(0)):
        # set the max number of tiles in `max_num`
        pixel_values = load_image(os.path.join(args.image_path, image_paths[i]), max_num=12).to(torch.bfloat16).to(model.device)
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        # single-image single-round conversation (单图单轮对话)
        # question = '<image>\nYou are doing the image captioning task. Describe this image in no more than 12 words.'
        question = '<image>\nYou are doing the image captioning task. Describe this image in one short sentence only.'
        # question = "<image>You are doing the VQA task, please answer me the question in one word or short phrase. Question: How many people are there in this picture? Answer: "
        # question = '<image>\nYou are doing the image captioning task. Given an image could be adversarially perturbed, describe this image in one short sentence only.'
        # question = '<image>\nYou are doing the feature extracting task. List the first three objects you note, if there are, in few words.'
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        print(f'User: {question}\nAssistant: {response}')

        img_path = image_paths[i]
        img_name = os.path.basename(img_path)
        match = re.search(r"\d{12}", img_name)
        if not match:
            raise ValueError(f"Invalid image name: {img_name}")
        image_id = int(match.group())

        generated_results.append({
            "image_id": image_id,
            "caption": response
        })

        if image_id not in reference_dict:
            reference_dict[image_id] = []
        reference_dict[image_id].extend(ref_captions_list[i])

# 保存生成结果
with open(adv_json_path, 'w') as f:
    json.dump(generated_results, f, indent=2)
print(f"\nGenerated captions saved to {adv_json_path}")

# 构建并保存参考数据
ref_data = {
    "images": [],
    "annotations": [],
    "type": "captions",
    "licenses": [],
    "info": {}
}
ann_id = 0
for image_id, captions in reference_dict.items():
    ref_data["images"].append({
        "id": image_id,
        "file_name": f"COCO_val2014_{image_id:012d}.jpg"
    })
    for caption in captions:
        ref_data["annotations"].append({
            "id": ann_id,
            "image_id": image_id,
            "caption": caption.strip()
        })
        ann_id += 1

with open(clean_json_path, 'w') as f:
    json.dump(ref_data, f, indent=2)
