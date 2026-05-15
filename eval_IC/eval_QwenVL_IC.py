import argparse as _argparse
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


def _exit_if_help_requested():
    if not any(arg in ("-h", "--help") for arg in _sys.argv[1:]):
        return
    parser = _argparse.ArgumentParser(description='image-captioning evaluation for Qwen2.5-VL.')
    parser.add_argument("--data_path", default="./data_annotation/coco_test_sub.json")
    parser.add_argument("--image_path", default="./adv_output/FOA")
    parser.add_argument("--output_dir", default="./caption_outputs")
    parser.add_argument("--model_path", default='Qwen/Qwen2.5-VL-7B-Instruct')
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

import os
import sys
import re
import datetime
import json
from tqdm import tqdm
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info




from dataset import paired_dataset2
args = _parse_runtime_args()
if not args.model_path:
    args.model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
if not args.cuda_visible_devices:
    args.cuda_visible_devices = "0,1,2"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
device = 'cuda'
# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", attn_implementation="sdpa", device_map="auto"
# )
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype="auto",
    # quantization_config=quant_config,
    attn_implementation="sdpa",
    device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained(args.model_path)


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
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/MIA')
# test_dataset = paired_dataset2('./data_annotation/coco_test_sub.json', s_test_transform, './adv_output/FS')

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
    
    for i in range(images.size(0)):
        messages = [
            # {"role": "system", "content": "You are an expert visual-language assistant. The input images may contain noise, occlusion, or other visual disturbances. Your task is to robustly perceive and reason over the visual content, identify meaningful elements, and provide accurate and coherent answers. Focus on core objects, actions, and relationships, and avoid being misled by irrelevant or corrupted regions. Use your knowledge to infer uncertain areas when needed."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(args.image_path, image_paths[i]),
                    },
                    {"type": "text", "text": "You are doing the image captioning task. Describe this image in one short sentence only."},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
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
