import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import os
import sys
import re
import warnings
import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm

from dataset import paired_dataset2

torch.manual_seed(100)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
device = 'cuda'
model = AutoModel.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

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
    images = images.to(dtype=torch.float16).to(device)
    
    for i in range(images.size(0)):
        image = Image.open(os.path.join('./adv_output/FOA',image_paths[i])).convert('RGB')
        # First round chat 
        question = "You are doing the image captioning task. Describe this image in one short sentence only."
        msgs = [{'role': 'user', 'content': [image, question]}]

        response = model.chat(
            msgs=msgs,
            tokenizer=tokenizer
        )
        print(response)


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