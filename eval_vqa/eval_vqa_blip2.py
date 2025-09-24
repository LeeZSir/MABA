import argparse
import os
import re
import random
# import gradio as gr
from ruamel.yaml import YAML

yaml=YAML(typ='safe')
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader,Subset

from transformers import BertForMaskedLM
from torchvision import transforms
from PIL import Image
from transformers import StoppingCriteriaList

from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import clip
from models.blip_model.blip_retrieval import blip_retrieval
# BLIP-2
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

import utils
import copy
import time


from dataset import paired_dataset2

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader


import tempfile

import json
import os

ds_collections = {
    'vqav2_val': {
        'train': './data_annotation/vqav2/vqav2_train.jsonl',
        'test': './data_annotation/vqav2/vqav2_val_subset.jsonl',
        'question': './data_annotation/vqav2/v2_OpenEnded_mscoco_val2014_questions_subset.json',
        'annotation': './data_annotation/vqav2/v2_mscoco_val2014_annotations_subset.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    }
}

class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot):
        self.test = open(test).readlines()
        self.prompt = prompt
        self.few_shot = few_shot

        if few_shot > 0:
            self.train = open(train).readlines()

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image = data['image']  # image path relative to root
        question = data['question']
        question_id = data['question_id']
        annotation = data.get('answer', None)

        few_shot_prompt = ''
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                sample = json.loads(sample.strip())
                few_shot_prompt += self.prompt.format(
                    sample['image'],
                    sample['question']) + f" {sample['answer']}\n"

        full_question = "Question: " + question + "Answer:"

        return {
            'image': image,  # you can join the base path later in the collate_fn
            'question': full_question,
            'question_id': question_id,
            'annotation': annotation
        }

def collate_fn(batch):
    image_paths = [item['image'] for item in batch]
    prompts = [item['question'] for item in batch]
    question_ids = [item['question_id'] for item in batch]
    annotations = [item['annotation'] for item in batch]
    return image_paths, prompts, question_ids, annotations


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='vqav2_val')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    device = torch.device('cuda:1')
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
    )

    prompt = '<img>{}</img>{} Answer:'

    dataset = VQADataset(
        train=ds_collections[args.dataset]['train'],
        test=ds_collections[args.dataset]['test'],
        prompt=prompt,
        few_shot=args.few_shot,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        # sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    outputs = []
    for _,batch in tqdm(enumerate(dataloader)):
        image_paths, prompts, question_ids, annotations = batch
        answers = []
        for i in range(len(image_paths)):
            # 原始路径
            image_path = image_paths[i]
            # 你希望拼接的前缀路径
            # prefix_path = "./datasets_own/MSCOCO/val2014"
            # prefix_path = "./adv_output/AttackVLM/val2014"
            # prefix_path = "./adv_output/AnyAttack/val2014"
            prefix_path = "./adv_output/FOA/val2014"
            # prefix_path = "./adv_output/TYAFCQformer-ITC-0.362.5-11-17Layers/val2014"
            # 提取文件名
            filename = os.path.basename(image_path)
            # 拼接新路径
            image_path = os.path.join(prefix_path, filename)
            raw_image = Image.open(image_path).convert('RGB')
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            # First round chat 
            question = prompts[i]
            answer = model.generate({"image": image, "prompt": question})[0]
            print(answer)
            answers.append(answer)

        for question_id, answer, annotation in zip(question_ids, answers,
                                                   annotations):
            if args.dataset in ['vqav2_val', 'vqav2_testdev', 'okvqa_val', 'textvqa_val', 'vizwiz_val']:
                outputs.append({
                    'question_id': question_id,
                    'answer': answer,
                })
            elif args.dataset in ['docvqa_val', 'infographicsvqa', 'gqa_testdev', 'ocrvqa_val', 'ocrvqa_test']:
                outputs.append({
                    'questionId': question_id,
                    'answer': answer,
                    'annotation': annotation,
                })
            elif args.dataset in ['ai2diagram_test']:
                outputs.append({
                    'image': question_id,
                    'answer': answer,
                    'annotation': annotation,
                })
            elif args.dataset in ['chartqa_test_human', 'chartqa_test_augmented']:
                outputs.append({
                    'answer': answer,
                    'annotation': annotation,
                })
            elif args.dataset in ['docvqa_test']:
                outputs.append({
                    'questionId': question_id,
                    'answer': answer,
                })
            elif args.dataset in ['vizwiz_test']:
                outputs.append({
                    'image': question_id,
                    'answer': answer,
                })
            else:
                raise NotImplementedError

    merged_outputs = outputs
    print(f"Evaluating {args.dataset} ...")
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    output_dir = './vqa_outputs'
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

    results_file = os.path.join(
        output_dir,
        f'{args.dataset}_{time_prefix}_fs{args.few_shot}_s{args.seed}.json'
    )
    json.dump(merged_outputs, open(results_file, 'w'), ensure_ascii=False)
