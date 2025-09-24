import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import copy
import math
import random
import sys
import warnings
import time
import datetime
from pathlib import Path
import argparse
import numpy as np

from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer



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

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def eval_model(args):
    # Data
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        idx = line["sample_id"]
        question_type = line["metadata"]["question_type"]
        dataset_name = line["metadata"]["dataset"]
        gt = line["conversations"][1]["value"]

        image_files = line["image"]
        qs = line["conversations"][0]["value"]
        cur_prompt = args.extra_prompt + qs

        args.conv_mode = "qwen_1_5"

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = preprocess_qwen([line["conversations"][0],{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
        img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)

        image_tensors = []
        for image_file in image_files:
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensors.append(image_tensor.half().cuda())
        # image_tensors = torch.cat(image_tensors, dim=0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
                                   "dataset": dataset_name,
                                   "sample_id": idx,
                                   "prompt": cur_prompt,
                                   "pred_response": outputs,
                                   "gt_response": gt,
                                   "shortuuid": ans_id,
                                   "model_id": model_name,
                                   "question_type": question_type,
                                   }) + "\n")
        ans_file.flush()

        if len(line["conversations"]) > 2:

            for i in range(2, len(line["conversations"]), 2):
                input_ids = torch.cat((input_ids, output_ids), dim=1)

                gt = line["conversations"][i + 1]["value"]
                qs = line["conversations"][i]["value"]
                cur_prompt = args.extra_prompt + qs

                args.conv_mode = "qwen_1_5"

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids_new = preprocess_qwen([line["conversations"][i],{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
                input_ids = torch.cat((input_ids, input_ids_new), dim=1)
                img_num = list(input_ids_new.squeeze()).count(IMAGE_TOKEN_INDEX)

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensors,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        use_cache=True)
        
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()

                ans_id = shortuuid.uuid()
                ans_file.write(json.dumps({
                                        "dataset": dataset_name,
                                        "sample_id": idx,
                                        "prompt": cur_prompt,
                                        "pred_response": outputs,
                                        "gt_response": gt,
                                        "shortuuid": ans_id,
                                        "model_id": model_name,
                                        "question_type": question_type,
                                        }) + "\n")
                ans_file.flush()


    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-si")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='vqav2_val')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Model
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-si"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto" # 显式指定模型所有权重到 cuda:1
    llava_model_args = {
        "multimodal": True,
        "attn_implementation": "sdpa",
    }
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, load_in_8bit = True,device_map=device_map, **llava_model_args)  # Add any other thing you want to pass in llava_model_args

    model.eval()
    prompt = '<img>{}</img>{} Answer:'
    dataset = VQADataset(
        train=ds_collections[args.dataset]['train'],
        test=ds_collections[args.dataset]['test'],
        prompt=prompt,
        few_shot=args.few_shot,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
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
            image = Image.open(image_path).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
            conv_template = "qwen_2"  # Make sure you use correct chat template for different models
            question = DEFAULT_IMAGE_TOKEN + "\nYou're doing the image VQA task, please answer me the following question in a word or short phrase." + prompts[i]
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
                max_new_tokens=1024,
            )
            answer = tokenizer.batch_decode(cont, skip_special_tokens=True)
            print(answer[0])
            answers.append(answer[0])

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

