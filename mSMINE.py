import argparse
import os
import random
import time
import numpy as np
import sys
import math
from PIL import Image
import torch.nn as nn
from torchvision.utils import save_image
from typing import Dict, List, Tuple, Optional, Set, Union, Sequence

import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import functional as _F
from torchvision.transforms import InterpolationMode
import numbers
import warnings
from ruamel.yaml import YAML
yaml=YAML(typ='safe')

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import clip
from transformers import BertForMaskedLM

import utils
from dataset import paired_dataset2
class maxSMI(nn.Module):
    '''
    critic : DV_Potential Func
    u_func : Fuction used to calculate the score of examples
    K : Nums of negative Examples
    '''
    def __init__(self,
         critic: nn.Module, 
         u_func: Optional[nn.Module] = None,
         K: Optional[int] = None,
         args: Optional[Dict] = None) -> None:
        
        super(maxSMI,self).__init__()
        self.critic = critic
        self.K = K
        self.u_func = u_func
    def forward(self, x, y, y0, txt2img, match_num=None,  K=None, only_it_loss=False):
        
        '''
        x:    [num_images, Embeddings]
        y:    [nums_texts, Embeddings]
        sX: Decomposed Slicing matrix for x
        sY: Decomposed Slicing matrix for y
        y0:   negative y [num_images, K, Embeddings]
        '''
        if K is None:
            K = self.K 
        g, g0_logsumexp, It_Loss, output  = self.mSMI(x, y, y0, txt2img, match_num, K)
        if only_it_loss:
            return torch.tensor(0.),torch.tensor(0.),It_Loss,torch.tensor(0.)
        g = g.detach()
        g0_logsumexp = g0_logsumexp.detach()
        It_Loss = It_Loss.detach()
        return g, g0_logsumexp, torch.tensor(0.), output

    def mSMI(self, x, y, y0, txt2img, match_num=None, K=None, only_it_loss=False):
        """
        Calculate the mSMI loss
        Parameters:
        -----------
        x:    [num_images, embed_dim]  # Image Embeddings
        y:    [num_texts, embed_dim]   # Text Embeddings
        y0:   [num_images, negative_examples, embed_dim]  # Negative Text Embeddings for corresponding images
        K:    int, optional  # Num of negative examples
        match_num: int, optional  # One image matches match_num texts(It means that, for one image, there are match_num pairs of positive samples)
        """
       # Get the number of images, which is also the number of batch_size
        num_images = x.shape[0]
        scales = x.shape[1]
        # Initialize variables
        g = 0.0
        g0_logmeanexp = 0.0
        # Calculating positive sample scores
        g_list = []
        # For each image, calculate the critic output and add it to the list.
        for i in range(num_images):
            # Get the scaled samples of the same image
            x_i = x[i, :, :] 
            # Get Positive text
            start_idx = i * match_num
            end_idx = (i + 1) * match_num
            # Get the texts for the current image
            y_i = y[start_idx:end_idx]   
            # Calculate the critic output  
            _, _, g_i = self.critic(x_i, y_i)  
            g_list.append(g_i)
        # For match_num pairs of positive samples, the average score is calculated cosidering to the DV_Potential Formula.
        # Vanilla
        g = torch.stack(g_list, dim=0).mean()
     
        if K is not None and y0 is not None:
            # Calculating negative sample scores
            g0_list = []
            # For each image, get the corresponding negative samples, calculate the critic output and add it to the list.
            for i in range(num_images):  
                # Get the scaled samples of the same image
                x_i = x[i, :, :]
                # Get the negative samples for the current image
                y0_i = y0[i, :, :]  
                _, _, g0_i = self.critic(x_i, y0_i)  # 计算当前负样本的 critic 输出
                g0_list.append(g0_i)
            # Calculate the logsumexp of neative samples
            # standard
            g0 = torch.stack(g0_list, dim=0)  # [num_images, K]
            g0_logmeanexp = torch.logsumexp(g0, dim=(0, 1, 2)) - math.log(num_images * scales * g0.shape[2])
            output = g - g0_logmeanexp
        else:
            # Calculate the mSMI loss when there is no negative samples
            output = g  
        return g, -g0_logmeanexp, torch.tensor(0.), output 
        

class BilinearCritic(nn.Module):
    '''
    encoder_x : dx -> feature_dim    Input:QR Decomposed Image() CLS Token [nums,256] @ Direction(Slicing) Matrix
    encoder_y : dy -> feature_dim    Input:QR Decomposed Text CLS Token  [nums,256] @ Direction(Slicing) Matrix
    u_func : 2*feature_dim -> 1 Output: Same Shape Tensors with the si_matrix
    tau: temperature
    '''
    
    def __init__(self,
                 encoder_x: nn.Module,
                 encoder_y: nn.Module,
                 u_func: nn.Module,
                 tau: Optional[float] = 1.):
        
        super(BilinearCritic,self).__init__()
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        self.u_func = u_func
        self.tau = torch.nn.Parameter(torch.Tensor([tau]))
        
    
    def forward(self, x, y, tau=None):
        if tau is None:
            tau = self.tau
        tau = torch.sqrt(tau)
        hx = self.norm(x)
        hy = self.norm(y)
        u = self.u_func(hx,hy.T)# + self.u_func(self.norm(self.encoder_x(x)),(self.norm(self.encoder_y(y)).T))
        
        return hx/tau, hy/tau, u
    
    def norm(self,z):
        # return z  
        return torch.nn.functional.normalize(z,dim=1)   
       
# Load Model
def load_model(args,model_name,text_encoder, device):         
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)    
    ### load checkpoint
    if model_name in ['CLIP_ViT-L/14']:
        model_name = 'ViT-L/14'
        model, preprocess = clip.load(model_name, device=device)
        model.set_tokenizer(tokenizer)
        return model, ref_model, tokenizer        
    else:
        model_name = 'ViT-B/16' if model_name == 'CLIP_ViT' else 'RN101'
        model, preprocess = clip.load(model_name, device=device)
        model.set_tokenizer(tokenizer)
    return model, ref_model, tokenizer

def get_scaled_imgs(imgs, scales=None, adv_imgs = None, device='cuda'):
    """
    对输入图像生成多个尺度版本，并将每个样本的原图与缩放图像存放在一起。
    
    输入:
        imgs: 张量，形状为 [B, C, H, W]
        scales: 缩放因子列表，例如 [0.5, 0.75, 1.25]。如果为 None，则只返回原图（但会增加一个尺度维度）。
        device: 噪声所在设备，如 'cuda' 或 'cpu'
        
    输出:
        输出张量形状为 [B, scales_num+1, C, H, W]，其中第 1 维（dim=1）的切片分别对应原图及各个缩放版本。
        例如，output[0, :, :, :, :] 就是第一张图片的所有尺度版本（第一个切片为原图，其余为各个缩放版本）。
    """
    # 如果 scales 为 None，则直接在尺度维度上增加一维返回原图
    if scales is None:
        return imgs.unsqueeze(1)  # 形状 [B, 1, C, H, W]
    # save_image(imgs[0],"ori.png")
    # 原图尺寸
    ori_shape = (imgs.shape[-2], imgs.shape[-1])
    # 定义一个转换，将缩放后的图像恢复到原始尺寸
    reverse_transform = transforms.Resize(ori_shape, interpolation=transforms.InterpolationMode.BICUBIC)
    Erasing_transform = transforms.RandomErasing(
        p=0.5,              
        scale=(0.1, 0.3), # 擦除区域面积占 2% 到 33%
        ratio=(1, 2),   # 宽高比在 0.3 到 3.3 之间
        value= 0 ,     # 用随机值填充擦除区域
        inplace= False       # 返回新的张量
    )
    Mask_transform = Random_Mask_Image(
        size=16,                  # 16x16的掩码块
        ratio=0.3,                # 40%覆盖率
        p=1,                    # 80%概率应用
        interpolation=InterpolationMode.BILINEAR,  # 预留插值模式
        inplace=False             # 安全模式（默认）
    )
    Rotation_transform = transforms. RandomRotation(30)
    scaled_imgs_list = []  # 用于存放每个缩放比例生成的图像版本
    for ratio in scales:
        # 根据缩放因子计算新的尺寸
        scale_shape = (int(ratio * ori_shape[0]), int(ratio * ori_shape[1]))
        scale_transform = transforms.Resize(scale_shape, interpolation=transforms.InterpolationMode.BICUBIC)
        
        # 给原图加上噪声（各尺度可以使用不同噪声，也可以根据需要调整）
        noise = torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
        temp_imgs = imgs + noise
        temp_imgs = Erasing_transform(temp_imgs)
        # 对加噪后的图像进行缩放
        temp_imgs = scale_transform(temp_imgs)
        # 限制像素值范围在 [0, 1]
        temp_imgs = torch.clamp(temp_imgs, 0.0, 1.0)
        # 将缩放后的图像恢复到原始尺寸
        temp_imgs = reverse_transform(temp_imgs)
        scaled_imgs_list.append(temp_imgs)
    # 将原图与各个尺度的图像按新维度堆叠：
    # 每个缩放版本的形状均为 [B, C, H, W]，stack 后输出形状为 [B, scales_num+1, C, H, W]
    if adv_imgs is not None:
        scaled_imgs_list.append(adv_imgs)

    output = torch.stack([imgs] + scaled_imgs_list, dim=1)
    return output

def random_mask(input_ids, tokenizer, device, mask_prob=0.15):
    labels = input_ids.clone()
    # 创建需要保护的特殊token列表
    special_tokens = [
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.mask_token_id
    ]
    
    # 创建随机概率矩阵（与input_ids形状相同）
    rand = torch.rand(input_ids.shape, device=device)
    
    # 初始化mask矩阵
    mask = (rand < mask_prob) & (~torch.isin(input_ids, torch.tensor(special_tokens, device=device)))
    
    # 应用mask策略（80%替换为[MASK]，10%随机词，10%保持原词）
    mask_token = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size
    
    # 80%的概率替换为[MASK]
    masked_indices = mask & (torch.rand(input_ids.shape, device=device) < 0.8)
    input_ids[masked_indices] = mask_token
    
    # 10%的概率替换为随机词
    random_indices = mask & ~masked_indices & (torch.rand(input_ids.shape, device=device) < 0.5)
    input_ids[random_indices] = torch.randint(0, vocab_size, input_ids[random_indices].shape, device=device)
    
    # 剩下的10%保持原词（无需操作）
    
    # 对于非mask位置，将labels设为-100（损失计算时忽略）
    labels[~mask] = -100
    
    return input_ids, labels

def random_swap(input_ids, tokenizer, device, swap_prob=0.6):
    """
    对输入的token序列进行随机交换，以swap_prob的概率进行增强。

    参数:
    - input_ids: 输入的token ID序列 (batch_size, seq_length)
    - tokenizer: 分词器对象
    - device: 计算设备
    - swap_prob: 交换增强的概率，默认0.6

    返回:
    - 交换后的input_ids
    """
    batch_size, seq_length = input_ids.shape
    swapped_input_ids = input_ids.clone()

    # 生成每个样本是否执行交换的随机标记
    apply_swap = torch.rand(batch_size, device=device) < swap_prob

    for i in range(batch_size):
        if apply_swap[i]:  # 仅对满足概率的句子进行随机交换
            # 选取可交换的token索引（排除特殊token）
            special_tokens = {
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
                tokenizer.mask_token_id
            }
            valid_indices = [idx for idx in range(seq_length) if swapped_input_ids[i, idx].item() not in special_tokens]
            
            if len(valid_indices) > 1:
                idx1, idx2 = torch.randperm(len(valid_indices), device=device)[:2].tolist()
                # 交换两个位置的token
                pos1, pos2 = valid_indices[idx1], valid_indices[idx2]
                swapped_input_ids[i, pos1], swapped_input_ids[i, pos2] = swapped_input_ids[i, pos2], swapped_input_ids[i, pos1]

    return swapped_input_ids


def main(args, config):
    # test_mSMI(args, config)
    # exit(1)
    # set device
    torch.cuda.set_device(args.cuda_id)
    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Model ####
    # create NE
    print("Creating NE")

    encoder_x = None
    encoder_y = None
    u_func = torch.matmul
    critic = BilinearCritic(encoder_x, encoder_y, u_func)
    NE = maxSMI(critic)
    NE.to(device)
    # create source model
    print("Creating Source Model")
    model, ref_model, tokenizer = load_model(args,args.source_model,args.source_text_encoder, device)
    model.to(device)
    ref_model.to(device)
    #### Dataset ####
    print("Creating dataset")
    # normalize
    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    # transforms
    s_test_transform = None
    n_px = model.visual.input_resolution
    s_test_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),       
    ])
    # Log_setting
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_dir = os.path.join("./slicing_matrix/checkpoints", timestamp,args.log)
    os.makedirs(save_dir, exist_ok=True)  

    # load dataset
    test_dataset = paired_dataset2(config['test_file'], s_test_transform, config['image_root'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=4, collate_fn=test_dataset.collate_fn)
    # Load data
    num_text = len(test_loader.dataset.text)
    num_image = len(test_loader.dataset.ann)

    if args.source_model in ['ALBEF', 'TCL']:
        # Initialize learnable slicing matrices
        sX = nn.Parameter(torch.randn(config['embed_dim'], config['embed_dim']- 35, device=device))
    else:
        sX = nn.Parameter(torch.randn(model.visual.output_dim, model.visual.output_dim - 384, device=device, dtype=torch.float32))

    # Define optimizer (include slicing matrices)
    optimizer = torch.optim.Adam(list(NE.parameters()) + [sX], lr=args.lr)
    # 每 50 个 epoch 让学习率乘以 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)
    # CosLR
    num_epochs = config.get('num_epochs',100)
    # mSMI Parameters
    # 负样本数量 (后续可调整)
    num_neg_samples = 10
    # 图像对应文本数
    match_num = 5
    # scales
    if args.scales is not None:
        scales = [float(itm) for itm in args.scales.split(',')]
        print(scales)
    else:
        scales = None
    scales_num = len(scales) +1

    # prepare intermediate representations
    s_feat_dict = {}
    s_feat_dict['s_image_feats'] = torch.zeros(num_image, scales_num, model.visual.output_dim)
    s_feat_dict['s_text_feats'] = torch.zeros(num_text, model.visual.output_dim)
    for epoch in range(num_epochs):  # 现在代码被 epoch 训练循环包裹
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        total_loss = 0.0  # 重新初始化 loss
        total_positive_loss = 0.0  # 重新初始化 loss
        total_negative_loss = 0.0  # 重新初始化 loss
        total_It_loss = 0.0  # 重新初始化 loss
        for batch_idx, (images, texts_group, images_ids, text_ids_groups, _) in enumerate(test_loader):
            print(f'--------------------> batch:{batch_idx}/{len(test_loader)}')
            txt2img = []
            texts_ids = []
            texts = []
            all_text_ids = set(range(len(s_feat_dict['s_text_feats'])))  # 所有文本 ID
            negative_texts_ids = []
            for i in range(len(texts_group)):
                texts += texts_group[i]
                texts_ids += text_ids_groups[i]
                # for labels matrix
                txt2img += [i]*len(text_ids_groups[i])


            for i in range(len(texts_group)):
                # 计算负样本：从非匹配文本中随机采样
                non_matched_texts = list(set(texts_ids) - set(text_ids_groups[i]))  # 排除当前图像的文本
                negative_texts_ids.append(random.sample(non_matched_texts, num_neg_samples))  # 采样负例

            images = images.to(device)
            # 获取增强图像
            images = get_scaled_imgs(images, scales, None, device)
            norm_scaled_imgs = []
            # 获取 Embeddings 和 CLS Tokens
            with torch.no_grad():
                # 处理缩放图像
                for img_id in range(images.shape[0]):
                    images_norm = images_normalize(images[img_id])
                    norm_scaled_imgs.append(images_norm)
                norm_scaled_imgs = torch.stack(norm_scaled_imgs,dim=0)
                # 处理缩放图像
                s_norm_output_img = []
                s_output_text = []
                max_length = 30 if args.source_model in ['ALBEF', 'TCL'] else 77
                # 遮挡文本
                texts_input = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, 
                        return_tensors="pt").to(device)
                masked_input, labels = random_mask(texts_input.input_ids, tokenizer, device)
                texts_input.input_ids = masked_input
                # 再对增强后的输入应用随机交换，swap_prob默认值为0.6
                swapped_input = random_swap(texts_input.input_ids, tokenizer, device, swap_prob = 1)
                texts_input.input_ids = swapped_input

                if args.source_model in ['ALBEF', 'TCL']:
                    for img_id in range(images.shape[0]):
                        s_output_img = (model.inference_image(norm_scaled_imgs[img_id]))['image_feat']
                        s_norm_output_img.append(s_output_img)
                    s_norm_output_img = torch.stack(s_norm_output_img, dim=0)

                    s_output_txt = model.inference_text(texts_input)
                    # 更新 text CLS Token
                    s_feat_dict['s_text_feats'][texts_ids] = s_output_txt['text_feat'].cpu().detach()
                else:
                    for img_id in range(images.shape[0]):
                        s_output_img = (model.inference_image(norm_scaled_imgs[img_id]))['image_feat']
                        s_norm_output_img.append(s_output_img)
                    s_norm_output_img = torch.stack(s_norm_output_img, dim=0).float()
                    output = model.inference_text(texts_input)
                    # s_feat_dict['s_image_feats'][images_ids] = output['image_feat'].cpu().float().detach()
                    s_feat_dict['s_text_feats'][texts_ids] = output['text_feat'].cpu().float().detach()
            # **确保数据一致性**
            batch_image_cls = s_norm_output_img.to(device)  # (batch_size, embed_dim)
            batch_text_cls = s_feat_dict['s_text_feats'][texts_ids].to(device)    # (batch_size, embed_dim)
            # **获取负样本 batch**
            negative_batch_text_cls = torch.stack(
                [s_feat_dict['s_text_feats'][neg_ids].to(device) for neg_ids in negative_texts_ids], dim=0
            )  # 形状: (batch_size, num_neg_samples, embed_dim)
            # Compute QR decomposition (保证 Q 矩阵用于投影)
            Q_sX, _ = torch.linalg.qr(sX)
            # Q_sY, _ = torch.linalg.qr(sY)

            # Only slicing
            batch_image_cls_proj = torch.matmul(batch_image_cls, Q_sX)  # (batch_size, num_neg_samples, embed_dim)
            batch_text_cls_proj = batch_text_cls @ Q_sX
            negative_batch_text_cls_proj = torch.matmul(negative_batch_text_cls, Q_sX)  # (batch_size, num_neg_samples, embed_dim)
            optimizer.zero_grad()
            g, g0_logsumexp,It_loss,loss = NE(batch_image_cls_proj, batch_text_cls_proj, negative_batch_text_cls_proj, txt2img, match_num=match_num, K=num_neg_samples)
            
            # Backpropagation, stochastic gradient-ascent
            (-loss).backward()
            optimizer.step()

            # **Reapply QR decomposition to project sX and sY onto Stiefel manifold**
            with torch.no_grad():
                sX.copy_(torch.linalg.qr(sX)[0])
                # sY.copy_(torch.linalg.qr(sY)[0])

            total_loss += loss.item()
            total_It_loss += It_loss.item()
            total_positive_loss += g.item()
            total_negative_loss += g0_logsumexp.item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(test_loader)}, Loss: {loss.item():.4f},It_Loss: {It_loss.item():.4f},positive_Loss: {g.item():.4f}, negative_Loss: {g0_logsumexp.item():.4f}")
        # 更新学习率
        scheduler.step()
        # 计算 Loss
        avg_loss = total_loss / len(test_loader)
        avg_It_loss = total_It_loss / len(test_loader)
        avg_positive_loss = total_positive_loss / len(test_loader)
        avg_negative_loss = total_negative_loss / len(test_loader)

        # 记录 Loss
        loss_filename = f"Loss_{timestamp}_{epoch}_{avg_loss:.4f}.txt"
        loss_file_path = os.path.join(save_dir, loss_filename)
        with open(loss_file_path, 'w') as f:
            f.write(f"Epoch: {epoch+1}/{num_epochs}\n")
            f.write(f"Avg Loss: {avg_loss:.6f}\n")
            f.write(f"Avg It Loss: {avg_It_loss:.6f}\n")
            f.write(f"Avg Positive Loss: {avg_positive_loss:.6f}\n")
            f.write(f"Avg Negative Loss: {avg_negative_loss:.6f}\n")

        # 记录 sX 和 sY
        slicing_filename = f"slicing_matrices_epoch_{epoch+1}.txt"
        slicing_file_path = os.path.join(save_dir, slicing_filename)
        with open(slicing_file_path, 'w') as f:
            f.write("sX:\n")
            sX_str = np.array2string(sX.detach().cpu().numpy(), precision=4, separator=', ')
            f.write(sX_str)

        # 保存 NE 模型和切片矩阵
        torch.save(NE.state_dict(), os.path.join(save_dir, f'NE_epoch_{epoch+1}.pt'))
        torch.save({
            'epoch': epoch + 1,
            'sX': sX.detach().cpu(),
        }, os.path.join(save_dir, f'slicing_matrices_epoch_{epoch+1}.pt'))

        print(f"All files for epoch {epoch+1} saved in {save_dir}")
    print("Training complete!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_coco.yaml')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--cuda_id', default=0, type=int)
    parser.add_argument('--log', default=None, type = str)

    parser.add_argument('--source_model', default='CLIP_ViT', type=str)
    parser.add_argument('--source_text_encoder', default='./checkpoints/Bert', type=str)   
    parser.add_argument('--target_text_encoder', default='./checkpoints/Bert', type=str)
 
    parser.add_argument('--original_rank_index_path', default='./std_eval_idx/mscoco_sub')  
    parser.add_argument('--scales', type=str, default=' 0.5, 0.75, 1.25, 1.5')#

    parser.add_argument('--lr',type=float, default=2e-4)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'))

    main(args, config)