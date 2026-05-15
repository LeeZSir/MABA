import argparse as _argparse
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


def _exit_if_help_requested():
    if not any(arg in ("-h", "--help") for arg in _sys.argv[1:]):
        return
    parser = _argparse.ArgumentParser(description='image-captioning evaluation for BLIP-2.')
    parser.add_argument("--data_path", default="./data_annotation/coco_test_sub.json")
    parser.add_argument("--image_path", default="./adv_output/FOA")
    parser.add_argument("--output_dir", default="./caption_outputs")
    parser.add_argument("--model_path", default='blip2_opt/caption_coco_opt6.7b')
    parser.add_argument("--cuda_visible_devices", default='1')
    parser.print_help()
    raise SystemExit(0)


_exit_if_help_requested()

import argparse
import os
import re
import random
from ruamel.yaml import YAML

yaml=YAML(typ='safe')
import numpy as np
import datetime
import json

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from lavis.models import load_model_and_preprocess

import utils
from dataset import paired_dataset2

from tqdm import tqdm

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
@torch.no_grad()
def evaluate_caption_model(model, test_loader, device, output_dir="./caption_outputs/"):
    model.eval()
    generated_results = []
    reference_dict = {}

    # 创建带时间戳的目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # 定义文件路径
    adv_json_path = os.path.join(save_dir, "adv.json")
    clean_json_path = os.path.join(save_dir, "clean.json")

    # 处理数据
    for batch in tqdm(test_loader, desc="Evaluating"):
        images, ref_captions_list, image_ids, _, image_paths = batch
        images = images.to(device)
        
        for i in range(images.size(0)):
            # [原有处理逻辑保持不变]
            single_image = images[i].unsqueeze(0)
            generated_caption = model.generate({"image": single_image},max_length = 30)[0]
            img_path = image_paths[i]
            img_name = os.path.basename(img_path)
            match = re.search(r"\d{12}", img_name)
            if not match:
                raise ValueError(f"Invalid image name: {img_name}")
            image_id = int(match.group())
            print(generated_caption)
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
    print(f"Reference captions saved to {clean_json_path}")

    # 使用保存的文件进行评估
    coco = COCO(clean_json_path)
    coco_res = coco.loadRes(adv_json_path)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    # 打印评估结果
    print("\n===== Captioning Evaluation Metrics =====")
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.4f}")

    return coco_eval.eval

def eval_captions(clean_json_path,adv_json_path):
    # 使用保存的文件进行评估
    coco = COCO(clean_json_path)
    coco_res = coco.loadRes(adv_json_path)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    # 打印评估结果
    print("\n===== Captioning Evaluation Metrics =====")
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.4f}")

    return coco_eval.eval


def main(args, config):
    torch.cuda.set_device(args.cuda_id)
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    eval_captions('./caption_outputs/MiniCPMo2.6-TYAFC/clean.json','./caption_outputs/MiniCPMv2.6-TYAFC/adv.json')
    exit(1)


    # BLIP-2系列
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
    )
    # Instruct-BLIP系列
    # model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna13b", is_eval=True, device=device)
    test_dataset = paired_dataset2(config['test_file'], vis_processors['eval'], config['image_root'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            num_workers=4, collate_fn=test_dataset.collate_fn)


    evaluate_caption_model(model, test_loader, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_coco.yaml')
    parser.add_argument('--minigpt4_config', default='./configs/minigpt4_eval.yaml')  
    parser.add_argument('--model_config', default='./configs/minigpt4_vicuna0.yaml')      
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--cuda_id', default=0, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'))
    main(args, config)   
