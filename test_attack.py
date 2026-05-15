import argparse
import os
import sys
from ruamel.yaml import YAML

yaml=YAML(typ='safe')


def build_parser():
    parser = argparse.ArgumentParser(description="Generate MABA adversarial examples.")
    parser.add_argument('--config', default='./configs/Retrieval_coco.yaml')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--cuda_id', default=0, type=int)
    parser.add_argument('--log', default='MABA', type=str)
    parser.add_argument('--output_dir', default='./adv_output', type=str)
    parser.add_argument('--source_model', default='CLIP_ViT-L/14', type=str)
    parser.add_argument('--source_text_encoder', default='bert-base-uncased', type=str)
    parser.add_argument(
        '--slicing_matrix_path',
        default=None,
        type=str,
        help='Path to slicing_matrices_epoch_*.pt. If omitted, MABA_SLICING_MATRIX is used.',
    )
    parser.add_argument('--scales', type=str, default='0.5,0.75,1.25,1.5')
    return parser


if __name__ == '__main__' and any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    build_parser().print_help()
    raise SystemExit(0)

import numpy as np
import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from transformers import BertForMaskedLM
from torchvision import transforms
from PIL import Image

from models.tokenization_bert import BertTokenizer
from models import clip

import utils
from MABA import MABAttacker, MABImageAttacker, MABTextAttacker
from dataset import paired_dataset2

def adv_gen(model, ref_model, data_loader, tokenizer, device, args):
    model.to(device)
    ref_model.to(device)
    model.float()
    model.eval()
    ref_model.eval()

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if args.source_model in ['ALBEF', 'TCL']:
        max_length = 30
    else:
        max_length = 77
    # Initialize attacker
    img_attacker = MABImageAttacker(
        images_normalize,
        eps=16/255,
        steps=30,
        step_size=3/255,
        args=args,
        slicing_matrix_path=args.slicing_matrix_path,
    )
    txt_attacker = MABTextAttacker(ref_model, tokenizer, cls=False, max_length=max_length, number_perturbation=1, topk=10, threshold_pred_score=0.3)
    attacker = MABAttacker(model, img_attacker, txt_attacker, args)

    print('Prepare memory')


    if args.scales is not None:
        scales = [float(itm) for itm in args.scales.split(',')]
        print(scales)
    else:
        scales = None

    print('Start Attack.')
    # Set save path
    adv_output_dir = os.path.join(args.output_dir, args.log, 'val2014')
    os.makedirs(adv_output_dir, exist_ok=True)

    for batch_idx, (images, texts_group, _, text_ids_groups, image_path) in enumerate(data_loader):
        print(f'--------------------> batch:{batch_idx}/{len(data_loader)}')
        txt2img = []
        texts = []
        for i in range(len(texts_group)):
            texts += texts_group[i]
            txt2img += [i]*len(text_ids_groups[i])

        images = images.to(device)
        adv_images, _, _ = attacker.attack(images, texts, txt2img, device=device, max_length=max_length, scales=scales)

        # Save Adversarial Examples
        for i in range(len(images)):
            original_path = image_path[i]
            filename = os.path.basename(original_path) 
            target_path = os.path.join(adv_output_dir, filename)
            os.makedirs(os.path.dirname(target_path), exist_ok=True) 
            save_image(adv_images[i], target_path)

def load_model(args,model_name,text_encoder, device):
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)  
    if model_name == 'CLIP_ViT':
        model_name = 'ViT-B/16'
        print(model_name)
    elif model_name == 'CLIP_ViT32':
        model_name = 'ViT-B/32'
    elif model_name == 'CLIP_ViT-L/14':
        model_name = 'ViT-L/14'
    else:
        model_name = 'RN101'
    model, preprocess = clip.load(model_name, device=device)
    model.set_tokenizer(tokenizer)
    return model, ref_model, tokenizer
    

def load_config(config_path):
    config_path = os.path.expanduser(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f)
    for key in ("test_file", "image_root"):
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
    return config



def main(args, config):
    torch.cuda.set_device(args.cuda_id)
    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    print("Creating Source Model")
    model, ref_model, tokenizer = load_model(args,args.source_model,args.source_text_encoder, device)


    #### Dataset ####
    print("Creating dataset")
    # Transforms
    n_px = model.visual.input_resolution
    s_test_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),       
    ])
    test_dataset = paired_dataset2(config['test_file'], s_test_transform, config['image_root'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            num_workers=4, collate_fn=test_dataset.collate_fn)
    adv_gen(model, ref_model, test_loader, tokenizer, device, args)

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    main(args, config)    
