import argparse as _argparse
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


def _exit_if_help_requested():
    if not any(arg in ("-h", "--help") for arg in _sys.argv[1:]):
        return
    parser = _argparse.ArgumentParser(description='image-captioning evaluation for LLaVA-OneVision.')
    parser.add_argument("--data_path", default="./data_annotation/coco_test_sub.json")
    parser.add_argument("--image_path", default="./adv_output/FOA")
    parser.add_argument("--output_dir", default="./caption_outputs")
    parser.add_argument("--model_path", default='lmms-lab/llava-onevision-qwen2-7b-si')
    parser.add_argument("--cuda_visible_devices", default='auto')
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

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from torch.utils.data import DataLoader

from PIL import Image
import copy
import torch
from torchvision import transforms

import os
import sys
import re
import warnings
import datetime
import json
from tqdm import tqdm

from dataset import paired_dataset2

warnings.filterwarnings("ignore")
args = _parse_runtime_args()
if not args.model_path:
    args.model_path = "lmms-lab/llava-onevision-qwen2-7b-si"
if args.cuda_visible_devices and args.cuda_visible_devices != "auto":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
pretrained = args.model_path
model_name = "llava_qwen"
device = "cuda"
device_map = "auto" # 显式指定模型所有权重到 cuda:1
llava_model_args = {
    "multimodal": True,
    "attn_implementation": "sdpa",
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, 
# load_8bit=True,
 device_map=device_map, **llava_model_args)  # Add any other thing you want to pass in llava_model_args

model.eval()

# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)
# image_tensor = process_images([image], image_processor, model.config)
# image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

# conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
# question = DEFAULT_IMAGE_TOKEN + "\nYou're doing the image captioning task. Descibe the image in a short sentence."
# conv = copy.deepcopy(conv_templates[conv_template])
# conv.append_message(conv.roles[0], question)
# conv.append_message(conv.roles[1], None)
# prompt_question = conv.get_prompt()

# input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
# image_sizes = [image.size]


# cont = model.generate(
#     input_ids,
#     images=image_tensor,
#     image_sizes=image_sizes,
#     do_sample=False,
#     temperature=0,
#     max_new_tokens=4096,
# )
# text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
# print(text_outputs)
s_test_transform = transforms.Compose([
    transforms.Resize(384, interpolation=Image.BICUBIC),
    transforms.ToTensor(),       
])
# load data
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './datasets_own/MSCOCO')
# test_dataset = paired_dataset2('./data_annotation/208589.json', s_test_transform, './adv_output/208589_clean')
# test_dataset = paired_dataset2('./data_annotation/208589.json', s_test_transform, './adv_output/208589')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/AttackVLM')
# test_dataset = paired_dataset2('./data_annotation/208589.json', s_test_transform, './adv_output/AttackVLM_00003')
# test_dataset = paired_dataset2('./data_annotation/208589.json', s_test_transform, './adv_output/AnyAttack')
test_dataset = paired_dataset2(args.data_path, s_test_transform, args.image_path)
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/TYAFCQformer-ITC-0.362.5-11-17Layers')
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
    images = images.to(dtype=torch.float16).to(device)
    
    for i in range(images.size(0)):
        # [原有处理逻辑保持不变]
        image = Image.open(os.path.join(args.image_path, image_paths[i]))
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        conv_template = "qwen_2"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + "\nYou are doing the image captioning task. Describe this image in one short sentence only."
        # question = DEFAULT_IMAGE_TOKEN + "\nYou are doing the VQA task, please answer me the question in one word or short phrase. Question:What bird is this? Answer: "
        # question = DEFAULT_IMAGE_TOKEN + "\nDescribe this image in detail."
        # question = DEFAULT_IMAGE_TOKEN + "\nDescribe this image in one short sentence only."
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]


        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        generated_caption = tokenizer.batch_decode(cont, skip_special_tokens=True)
        print(generated_caption)
        img_path = image_paths[i]
        img_name = os.path.basename(img_path)
        match = re.search(r"\d{12}", img_name)
        if not match:
            raise ValueError(f"Invalid image name: {img_name}")
        image_id = int(match.group())

        generated_results.append({
            "image_id": image_id,
            "caption": generated_caption
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
