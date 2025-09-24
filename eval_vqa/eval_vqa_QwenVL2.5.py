import argparse
import itertools
import json
import os
import random
import time
from functools import partial
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from vqa import VQA
from vqa_eval import VQAEval

ds_collections = {
    'vqav2_val': {
        'train': './data_annotation/vqav2/vqav2_train.jsonl',
        'test': './data_annotation/vqav2/vqav2_val_subset.jsonl',
        'question': './data_annotation/vqav2/v2_OpenEnded_mscoco_val2014_questions_subset.json',
        'annotation': './data_annotation/vqav2/v2_mscoco_val2014_annotations_subset.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'data/okvqa/okvqa_train.jsonl',
        'test': 'data/okvqa/okvqa_val.jsonl',
        'question': 'data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_val.jsonl',
        'question': 'data/vizwiz/vizwiz_val_questions.json',
        'annotation': 'data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_test': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_test.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'docvqa_val': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/val.jsonl',
        'annotation': 'data/docvqa/val/val_v1.0.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'docvqa_test': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'train': 'data/chartqa/train_human.jsonl',
        'test': 'data/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'train': 'data/chartqa/train_augmented.jsonl',
        'test': 'data/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'gqa_testdev': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/testdev_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'ocrvqa_val': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'train': 'data/ai2diagram/train.jsonl',
        'test': 'data/ai2diagram/test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    }
}

# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


# def collate_fn(batches, tokenizer):

#     questions = [_['question'] for _ in batches]
#     question_ids = [_['question_id'] for _ in batches]
#     annotations = [_['annotation'] for _ in batches]

#     input_ids = tokenizer(questions, return_tensors='pt', padding='longest')

#     return question_ids, input_ids.input_ids, input_ids.attention_mask, annotations
import json
import random
import os
import torch

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


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # torch.distributed.init_process_group(
    #     backend='nccl',
    #     world_size=int(os.getenv('WORLD_SIZE', '1')),
    #     rank=int(os.getenv('RANK', '0')),
    # )

    # torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    # vqa = VQA(ds_collections[args.dataset]['annotation'],
    #             ds_collections[args.dataset]['question'])
    # resFile = './vqa_outputs/InternVL3-8B-AttackVLM-Newprompt.json'
    # results = vqa.loadRes(
    #     resFile=resFile,
    #     quesFile=ds_collections[args.dataset]['question'])
    # vqa_scorer = VQAEval(vqa, results, n=2)
    # vqa_scorer.evaluate()

    # # print(vqa_scorer.accuracy)
    # # # exit(1)
    # # # 原始评估结果
    # acc = vqa_scorer.accuracy  # 或者你可以直接赋值 acc = {...} 用于调试
    # # 提取 perQuestionType 部分并排序
    # pq_type = acc.get("perQuestionType", {})
    # sorted_pq = sorted(pq_type.items(), key=lambda x: x[1], reverse=True)
    # # 取前5名和后5名
    # top5 = sorted_pq[:5]
    # bottom5 = sorted_pq[-5:]
    # # 构造新的结果字典
    # result_with_detail = {
    #     "full_accuracy": acc,
    #     "top_5_question_types": [{"type": k, "score": v} for k, v in top5],
    #     "bottom_5_question_types": [{"type": k, "score": v} for k, v in bottom5],
    # }
    # # 生成输出路径
    # input_path = resFile
    # base, ext = os.path.splitext(input_path)
    # output_path = base + "-Detail" + ext
    # # 保存为 JSON 文件
    # with open(output_path, 'w') as f:
    #     json.dump(result_with_detail, f, indent=2)
    # exit(1)
    # from transformers import BitsAndBytesConfig

    # quant_config = BitsAndBytesConfig(
    #     load_in_8bit=True
    # )

    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct",
    #     # quantization_config=quant_config, 
    #     attn_implementation="sdpa", 
    #     torch_dtype="auto", 
    #     device_map="auto"
    # )
    # # default processor
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # print(f"保存成功: {output_path}")
    # 测算分type结果
    # 你手动提供的5个JSON路径
    json_files = [
        './vqa_outputs/InternVL3-8B-AttackVLM-Newprompt-Detail.json',
        './vqa_outputs/LLAVAOVsi-7B-AttackVLM-Detail.json',
        './vqa_outputs/MiniCPMo2.6-AttackVLM-Detail.json',
        './vqa_outputs/Phi4Multimodal-AttackVLM-Detail.json',
        './vqa_outputs/QwenVL2.5-7B-AttackVLM-Detail.json',
    ]

    # 累加器和计数器
    total_scores = {'yes/no': 0.0, 'number': 0.0, 'other': 0.0}
    counts = {'yes/no': 0, 'number': 0, 'other': 0}

    for path in json_files:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            answer_type = data['full_accuracy']['perAnswerType']

            for key in ['yes/no', 'number', 'other']:
                if key in answer_type:
                    total_scores[key] += answer_type[key]
                    counts[key] += 1
                else:
                    print(f"[警告] {path} 缺少字段 {key}")

        except Exception as e:
            print(f"[错误] 处理 {path} 时出错: {e}")

    # 打印最终平均值
    print("\n=== Answer Type 平均准确率 ===")
    for key in ['yes/no', 'number', 'other']:
        if counts[key] > 0:
            avg = total_scores[key] / counts[key]
            print(f"{key:<7}: {avg:.2f}")
        else:
            print(f"{key:<7}: 无数据")
    exit(1)

    prompt = '<img>{}</img>{} Answer:'


    random.seed(args.seed)
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
            # prefix_path = "./adv_output/TYAFCQformer-ITC-0.362.5-11-17Layers/val2014"
            # prefix_path = "./adv_output/MIA/val2014"
            # prefix_path = "./adv_output/FS/val2014"
            prefix_path = "./adv_output/TYAFC/val2014"
            # 提取文件名
            filename = os.path.basename(image_path)
            # 拼接新路径
            image_path = os.path.join(prefix_path, filename)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": "You are doing the VQA Task, please answer me the following question in a word or short phrase." + prompts[i]},
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
            answer = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
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

    # torch.distributed.barrier()

    # world_size = torch.distributed.get_world_size()
    # merged_outputs = [None for _ in range(world_size)]
    # torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    # merged_outputs = [json.loads(_) for _ in merged_outputs]
    # merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
    merged_outputs = outputs
    if True:
        print(f"Evaluating {args.dataset} ...")
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        output_dir = './vqa_outputs'
        os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

        results_file = os.path.join(
            output_dir,
            f'{args.dataset}_{time_prefix}_fs{args.few_shot}_s{args.seed}.json'
        )
        json.dump(merged_outputs, open(results_file, 'w'), ensure_ascii=False)

        if ds_collections[args.dataset]['metric'] == 'vqa_score':
            vqa = VQA(ds_collections[args.dataset]['annotation'],
                      ds_collections[args.dataset]['question'])
            results = vqa.loadRes(
                resFile=results_file,
                quesFile=ds_collections[args.dataset]['question'])
            vqa_scorer = VQAEval(vqa, results, n=2)
            vqa_scorer.evaluate()

            print(vqa_scorer.accuracy)

        elif ds_collections[args.dataset]['metric'] == 'anls':
            json.dump(merged_outputs,
                      open(results_file, 'w'),
                      ensure_ascii=False)
            print('python infographicsvqa_eval.py -g ' +
                  ds_collections[args.dataset]['annotation'] + ' -s ' +
                  results_file)
            os.system('python infographicsvqa_eval.py -g ' +
                      ds_collections[args.dataset]['annotation'] + ' -s ' +
                      results_file)
        elif ds_collections[args.dataset]['metric'] == 'relaxed_accuracy':
            print({
                'relaxed_accuracy': evaluate_relaxed_accuracy(merged_outputs)
            })
        elif ds_collections[args.dataset]['metric'] == 'accuracy':
            if 'gqa' in args.dataset:
                for entry in merged_outputs:
                    response = entry['answer']
                    response = response.strip().split('.')[0].split(
                        ',')[0].split('!')[0].lower()
                    if 'is ' in response:
                        response = response.split('is ')[1]
                    if 'are ' in response:
                        response = response.split('are ')[1]
                    if 'a ' in response:
                        response = response.split('a ')[1]
                    if 'an ' in response:
                        response = response.split('an ')[1]
                    if 'the ' in response:
                        response = response.split('the ')[1]
                    if ' of' in response:
                        response = response.split(' of')[0]
                    response = response.strip()
                    entry['answer'] = response
            print({'accuracy': evaluate_exact_match_accuracy(merged_outputs)})

    torch.distributed.barrier()
