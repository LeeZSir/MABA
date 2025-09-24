import os
import sys
import re
import warnings
import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm

import requests
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from dataset import paired_dataset2
# Set device to GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model path
model_path = "microsoft/Phi-4-multimodal-instruct"

# Load processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Load model on GPU 1 explicitly
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": 0},  # Now CUDA_VISIBLE_DEVICES=1, so ": 0" maps to GPU 1
    torch_dtype="auto",
    trust_remote_code=True,
    _attn_implementation='sdpa',
).to(device)

# Load generation config
generation_config = GenerationConfig.from_pretrained(model_path)

# Load dataset
s_test_transform = transforms.Compose([
    transforms.Resize(384, interpolation=Image.BICUBIC),
    transforms.ToTensor(),       
])
# load data
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './datasets_own/MSCOCO')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/AttackVLM')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/AnyAttack')
test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/FOA')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/TYAFCQformer-ITC-0.362.5-11-17Layers')
test_loader = DataLoader(test_dataset, batch_size=1,
                        num_workers=4, collate_fn=test_dataset.collate_fn)
generated_results = []
reference_dict = {}

# 创建带时间戳的目录
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join('./caption_outputs', timestamp)
os.makedirs(save_dir, exist_ok=True)

# 定义文件路径
adv_json_path = os.path.join(save_dir, "adv.json")
clean_json_path = os.path.join(save_dir, "clean.json")

# 处理数据
for batch in tqdm(test_loader, desc="Evaluating"):
    images, ref_captions_list, image_ids, _, image_paths = batch
    
    for i in range(images.size(0)):
        # Prompt structure
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        # Part 1: Image Processing
        print("\n--- IMAGE PROCESSING ---")
        prompt = f'{user_prompt}You are doing the image captioning task. Describe this <|image_1|> in one short sentence only. {prompt_suffix}{assistant_prompt}'
        # Load image
        image = Image.open(os.path.join('./adv_output/FOA',image_paths[i])).convert('RGB')
        # Process inputs and move to device
        inputs = processor(text=prompt, images=image, return_tensors='pt').to(device)
        # Generate output
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=generation_config,
        )
        # Extract generated part only
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        # Decode
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        print(f'>>> Response\n{response}')
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


