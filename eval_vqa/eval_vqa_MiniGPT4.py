import math
import argparse
import numpy as np
from PIL import Image
import random
import os
import sys
import re
import warnings
import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm

import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub,Conversation, SeparatorStyle

device = "cuda:2" if torch.cuda.is_available() else "cpu"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2023
conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

def setup_seeds_another(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
# ------------------------------------------------------------------ #

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

def initialize_model(cfg):
    # ========================================
    #             Model Initialization
    # ========================================
    setup_seeds_another(42)

    model_config = cfg.model_cfg
    model_config.do_sample = False
    # model_config.device_8bit = os.environ["CUDA_VISIBLE_DEVICES"]
    model_cls = registry.get_model_class(model_config.arch)
    # model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model = model_cls.from_config(model_config).to(device)
    if cfg.model_cfg.arch=='minigpt_v2':
        CONV_VISION = Conversation(
            system="",
            roles=(r"<s>[INST] ", r" [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )
    else:
        CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    # stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
    stop_words_ids = [torch.tensor(ids).to(device=device) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    # chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
    chat = Chat(model, vis_processor, device=device, stopping_criteria=stopping_criteria)
    print('Initialization Finished')

    return chat, CONV_VISION

def chat_with_image_path_and_question(chat, CONV_VISION, image_path, query):
    # ========================================
    #                Chatting
    # ========================================

    image = Image.open(image_path).convert('RGB')

    # print("Image loaded.")
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(image, chat_state, img_list)
    # print("Chat loaded.")
    chat.encode_img(img_list)
    # print("Chat encode_img.")
    chat.ask(query, chat_state)
    # print("Chat ask.")

    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              max_new_tokens=300,
                              max_length=2000)[0]
    print("Answer:", llm_message)
    return llm_message

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
    # minigpt-4
    parser.add_argument("--cfg_path", default="./configs/minigpt4_eval.yaml")
    parser.add_argument("--data_path", default="./data_annotation/coco_test_sub.json", type=str)
    parser.add_argument("--image_path", default="./adv_output/FOA", type=str)
    parser.add_argument("--gt_path", default="coco_karpathy_test_gt.json")
    parser.add_argument("--llama_path",default='./checkpoints/VicunaV013B')
    parser.add_argument("--ckpt_path",help="./checkpoints/VicunaV013B/pretrained_minigpt4.pth")
    parser.add_argument("--output_path", default="./vqa_outputs/MiniGPT4-TYAFC/adv.json", type=str)
    parser.add_argument("--prompt",default="Could you describe the contents of this image for me.",type=str)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    print("output_path:", args.output_path)
    print(f"Loading MiniGPT-4 models..")
    cfg = Config(args)
    if args.llama_path:
        cfg.config["model"]["llama_model"]=args.llama_path
    if args.ckpt_path:
        cfg.config["model"]["ckpt"]=args.ckpt_path

    chat, CONV_VISION=initialize_model(cfg)
    print(f"Done")

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
            prompt = prompts[i]
            answer = chat_with_image_path_and_question(chat, CONV_VISION, image_path, prompt)
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

