import numpy as np
import torch
import torch.nn as nn

import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import random
import time

from mSMINE import maxSMI,BilinearCritic



class TFCAttacker():
    def __init__(self, model, img_attacker, txt_attacker, args):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker
        self.args = args

    def attack(self, imgs, txts, txt2img, all_texts, device='cpu', max_length=30, scales=None, masks=None, **kwargs):

        with torch.no_grad():
            origin_img_output = self.model.inference_image(self.img_attacker.normalization(imgs))
            img_supervisions = origin_img_output['image_feat'][txt2img]
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions)
        # 每次换完词都清空一次
        # clear_text_hook_outputs()
        with torch.no_grad():
            txts_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True,
                                                     max_length=max_length, return_tensors="pt").to(device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output['text_feat']
            all_txt_supervisions = None

        start_time = time.time()
        adv_imgs, last_adv_imgs = self.img_attacker.txt_guided_attack(self.model, imgs, txt2img, all_txt_supervisions, device,
                                                                      scales=scales, txt_embeds=txt_supervisions, inter_text_embeddings = None)
        end_time = time.time()
        execuate_time = end_time - start_time
        # 清除中间文本hook
        # for h in text_hooks:
        #     h.remove()
        with torch.no_grad():
            adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(adv_imgs))
            adv_img_supervisions = adv_imgs_outputs['image_feat'][txt2img]
            last_adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(last_adv_imgs))
            last_adv_img_supervisions = last_adv_imgs_outputs['image_feat'][txt2img]
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions,
                                                       adv_img_embeds=adv_img_supervisions,
                                                       last_adv_img_embeds=last_adv_img_supervisions)
        return adv_imgs, adv_txts, execuate_time


class TFCImageAttacker():
    def __init__(self, normalization, eps=2 / 255, steps=10, step_size=0.217 / 255, sample_numbers=5,args = None):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size
        self.sample_numbers = sample_numbers
        # corpus
        # self.projection_matrix = torch.load('./checkpoints/20250221_181911/Corpus_matrix.pt', map_location='cpu')
        self.projection_matrix = None
        # mSMI
        # Load mSMI Matrix
        self.checkpoint_X = torch.load('./slicing_matrix/checkpoints/QformerL14-itonly-1000Samples/slicing_matrices_epoch_100.pt', map_location='cpu')
        self.sX = self.checkpoint_X['sX']
        # 只是用sX矩阵
        # self.checkpoint_Y = torch.load('./checkpoints/20250224_163718/slicing_matrices_epoch_80.pt', map_location='cpu')
        self.sY = self.sX
        # self.sY = self.checkpoint_X['sY']
        # Compute QR decomposition (保证 Q 矩阵用于投影)
        self.Q_sX, _ = torch.linalg.qr(self.sX)
        self.Q_sY, _ = torch.linalg.qr(self.sY)
        self.args = args
        # self.Q_sX = self.Q_sX + self.Q_sY * 0.45
    # Projection
    def loss_func(self, adv_imgs_embeds, txts_embeds, txt2img,all_txt_supervisions):
        """    
        all_txt_supervisions (torch.Tensor): The supervised text embeddings.

        Returns:
        loss (torch.Tensor): The loss for IATC.
        """
        device = adv_imgs_embeds.device
        # self.projection_matrix =self.projection_matrix.to(device)
        self.Q_sX = self.Q_sX.to(device)
        self.Q_sY = self.Q_sY.to(device)
        # corpus space
        adv_imgs_embeds = adv_imgs_embeds @ self.Q_sX
        txts_embeds = txts_embeds @ self.Q_sY
        # print(adv_imgs_embeds.shape)
        # print(txts_embeds.shape)
        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        # print(txt2img)
        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos

        return loss
    
    def loss_func_old(self, adv_imgs_embeds, txts_embeds, txt2img):  
        device = adv_imgs_embeds.device    

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        
        for i in range(len(txt2img)):
            it_labels[txt2img[i], i]=1
        
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos
        
        return loss

    def rand3Num(self): ### num1 -> adv num2-> clean num3->last
        while True:
            num1 = random.randint(1, 100)
            if 100 - num1 > 1:
                num2 = random.randint(1, 100 - num1)
            else:
                num1 = 98
                num2 = 1
            num3 = 100 - num1 - num2
            
            if 1 <= num3 <= 100 and num1 < num3 and num3 < num2:
                break

        return (num1, num2, num3)
    def generate_mid_centric_weights(self, num_layers, sigma=2.0):
        """
        生成高斯峰值在中间的层权重
        sigma: 控制权重集中程度（值越小，中间层权重越集中）
        """
        center = (num_layers - 1) / 2  # 中心位置（如11层时center=5.0）
        indices = torch.arange(num_layers, dtype=torch.float32)
        weights = torch.exp(-(indices - center)**2 / (2 * sigma**2))
        return weights / weights.sum()  # 归一化

    def generate_descending_weights(self,num_layers):
        weights = torch.arange(num_layers, 0, -1, dtype=torch.float32)  # 从num_layers到1
        return weights / weights.sum()

    def generate_ascending_weights(self,num_layers):
        weights = torch.arange(1, num_layers+1, dtype=torch.float32)  # 从1到num_layers
        return weights / weights.sum()

    def variance_based_weights(self, diversity_matrix, temp=0.2, eps=1e-6):
        """
        平滑版方差权重生成器：通过温度系数控制权重分布尖锐度
        Args:
            diversity_matrix: 形状 [Layer, Batch, Seq, Embed]
            temp (float): 温度系数（0~1），值越大权重分布越均匀
            eps: 数值稳定系数
        Returns:
            layer_weights: 形状 [Layer] 的平滑归一化权重
        """
        # 展平计算方差
        reshaped = diversity_matrix.view(diversity_matrix.size(0), -1)  # [Layer, ...]
        layer_vars = torch.var(reshaped, dim=1, unbiased=False)  # [Layer]
        
        # 对数变换压缩方差范围（+1防止对0取log）
        log_vars = torch.log(layer_vars + 1.0)
        
        # 应用温度缩放（Softmax温度系数原理）
        scaled_vars = log_vars / temp
        
        # Softmax归一化（方差大的层仍权重高，但分布更平滑）
        weights = torch.softmax(scaled_vars, dim=0)
        
        return weights
   
    def svd_enhanced_rank_constraint(self, matrix):
        """ 对每个layer-batch组合的[Seq, Embed]矩阵进行SVD约束，保留前k个奇异值，压缩剩余部分 """
        # 合并layer和batch维度
        L, B, S, E = matrix.shape
        reshaped = matrix.view(L*B, S, E)     
        # 批处理SVD
        U, S_vec, Vh = torch.linalg.svd(reshaped, full_matrices=False)
        k = int(S_vec.shape[-1] * 0.04)
        # 核范数计算（仅压缩后k个奇异值）
        nuclear_loss = S_vec[:, :40].sum(dim=-1)# + S_vec[:, k:].sum(dim=-1) * 0.1 # 形状 [L*B]
        return nuclear_loss.view(L, B)  # 恢复为 [layer, batch]



    def txt_guided_attack(self, model, imgs, txt2img, all_txt_supervisions,device, scales=None, txt_embeds=None,inter_text_embeddings = None):

        # 用于存储不同层的中间特征
        visual_self_attn_outputs = {i: [] for i in list(range(11,17)) + [23]}

        def create_visual_hook(layer_idx):
            def hook_fn(module, input, output):
                visual_self_attn_outputs[layer_idx].append(output)
            return hook_fn

        # 注册 Hook
        visual_hooks = []
        for i in list(range(11,17)) + [23]:
            if self.args.source_model in ['ALBEF', 'TCL']:
                v_handle = model.visual_encoder.blocks[i].norm2.register_forward_hook(create_visual_hook(i))
            elif self.args.source_model in ['CLIP_ViT','CLIP_ViT-L/14']:
                v_handle = model.visual.transformer.resblocks[i].ln_2.register_forward_hook(create_visual_hook(i))
            elif self.args.source_model in ['CLIP_CNN']:
                v_handle = model.visual.layer4[2].bn3.register_forward_hook(create_visual_hook(i))
            visual_hooks.append(v_handle)

        def clear_visual_hook_outputs():
            for key in visual_self_attn_outputs:
                visual_self_attn_outputs[key] = []

        model.eval()

        b, _, _, _ = imgs.shape

        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) + 1

        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(
            device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        last_adv_imgs = None

        start_time = time.time()
        ratio_list = []

        for step in range(self.steps):  # self.steps=10
            if last_adv_imgs != None:
                samples = []
                clone_adv_imgs = adv_imgs.clone()
                loss_list = []
                for k in range(self.sample_numbers):
                    samples.append(self.rand3Num())
                for sample in samples:
                    adv_imgs = (sample[0] / 100) * clone_adv_imgs + (sample[1] / 100) * imgs + (
                                sample[2] / 100) * last_adv_imgs
                    adv_imgs.requires_grad_()

                    if self.normalization is not None:
                        adv_imgs_output = model.inference_image(self.normalization(adv_imgs))
                    else:
                        adv_imgs_output = model.inference_image(adv_imgs)

                    adv_imgs_embeds = adv_imgs_output['image_feat']
                    model.zero_grad()
                    with torch.enable_grad():
                        loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                        loss = self.loss_func(adv_imgs_embeds, txt_embeds, txt2img,all_txt_supervisions)
                    loss.backward()
                    grad = adv_imgs.grad
                    grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                    perturbation = self.step_size * grad.sign()

                    adv_imgs = clone_adv_imgs.detach() + perturbation
                    adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                    adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

                    if self.normalization is not None:
                        adv_imgs_output = model.inference_image(self.normalization(adv_imgs))
                    else:
                        adv_imgs_output = model.inference_image(adv_imgs)
                    adv_imgs_embeds = adv_imgs_output['image_feat']
                    model.zero_grad()
                    with torch.enable_grad():
                        loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                        loss = self.loss_func(adv_imgs_embeds, txt_embeds, txt2img,all_txt_supervisions)
                    loss.backward()
                    loss_list.append(loss.item())
                #candidate_index = loss_list.index(max(loss_list))

                candidate_index = loss_list.index(max(loss_list))
                ratio_list.append(samples[candidate_index])

                adv_imgs = (samples[candidate_index][0] / 100) * clone_adv_imgs + (
                            samples[candidate_index][1] / 100) * imgs + (
                                       samples[candidate_index][2] / 100) * last_adv_imgs
                adv_imgs.requires_grad_()
                scaled_imgs = self.get_scaled_imgs(adv_imgs, [0.5, 0.75, 1.25, 1.5], device)
            # --- 核心梯度投影逻辑 ---
                model.zero_grad()
                with torch.enable_grad():
                    clear_visual_hook_outputs()
                    adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))
                    # 为最后一张图像获取中间输出    
                    if self.normalization is not None:
                        if self.args.source_model in ['ALBEF', 'TCL']:
                            inter_visual_embeddings = []
                            for layer_idx in sorted(visual_self_attn_outputs.keys()):
                                layer_embed = torch.stack(visual_self_attn_outputs[layer_idx], dim=0).squeeze(0)  # [batch, seq, embed]
                                inter_visual_embeddings.append(layer_embed)
                            inter_visual_embeddings = torch.stack(inter_visual_embeddings, dim=0)  # [layers, batch, seq, embed]
                        elif self.args.source_model in ['CLIP_ViT','CLIP_ViT-L/14']:
                            inter_visual_embeddings = []
                            for layer_idx in sorted(visual_self_attn_outputs.keys()):
                                layer_embed = torch.stack(visual_self_attn_outputs[layer_idx], dim=0).squeeze(0).permute(1, 0, 2)  # [batch, seq, embed]
                                inter_visual_embeddings.append(layer_embed)
                            inter_visual_embeddings = torch.stack(inter_visual_embeddings, dim=0)  # [layers, batch, seq, embed]                        
                    else:
                        adv_imgs_output = model.inference_image(scaled_imgs)
                        if self.args.source_model in ['ALBEF', 'TCL']:
                            inter_visual_embeddings = []
                            for layer_idx in sorted(visual_self_attn_outputs.keys()):
                                layer_embed = torch.stack(visual_self_attn_outputs[layer_idx], dim=0).squeeze(0)
                                inter_visual_embeddings.append(layer_embed)
                            inter_visual_embeddings = torch.stack(inter_visual_embeddings, dim=0)
                        elif self.args.source_model in ['CLIP_ViT','CLIP_ViT-L/14']:
                            inter_visual_embeddings = []
                            for layer_idx in sorted(visual_self_attn_outputs.keys()):
                                torch.stack(visual_self_attn_outputs[layer_idx], dim=0).squeeze(0).permute(1, 0, 2)
                                inter_visual_embeddings.append(layer_embed)
                            inter_visual_embeddings = torch.stack(inter_visual_embeddings, dim=0)                          
                    adv_embeds = adv_imgs_output['image_feat']
                    # 计算SA_loss（对抗损失）
                    SA_loss = torch.tensor(0.0, device=device)
                    for i in range(5):
                        SA_loss += self.loss_func(adv_embeds[i*b:(i+1)*b], txt_embeds, txt2img, all_txt_supervisions)
                    # print(f"sa_loss: {SA_loss}")
                    # --- 梯度投影 ---
                    # 计算SA_loss梯度
                    SA_loss.backward(retain_graph=True)
                    adv_imgs.retain_grad()
                    sa_grad = adv_imgs.grad.detach().clone()
                    adv_imgs.grad.zero_()
                    # 清除之前的梯度
                    model.zero_grad()
                   # 计算FC_loss（正则项）
                    num_layers = inter_visual_embeddings.shape[0] - 1 #减掉最后一层
                    inter_shape_x = inter_visual_embeddings.shape[2]
                    one_matrix = torch.ones((inter_shape_x, 1)).to(device)
                    rank1_approx = inter_visual_embeddings[:-1,:,:,:].sum(dim=2, keepdim=True) / inter_shape_x
                    Diversity_Matrix = inter_visual_embeddings[:-1,:,:,:] - one_matrix @ rank1_approx# 
                    # nuclear_norm = self.svd_enhanced_rank_constraint(Diversity_Matrix,inter_visual_embeddings[-1][:,0,:]) #传入最后一层CLS Token
                    # frobenius_norm = torch.norm(Diversity_Matrix.mean(dim=1), p='fro',dim=(1,2))
                    # Diversity_Matrix_CLS = F.normalize(model.vision_proj(Diversity_Matrix.mean(dim=2,keepdim=True)),dim=-1).cpu().to(device)
                    # proj_Diversity_Matrix_CLS = Diversity_Matrix_CLS @ self.Q_sX # Proj版
                    # nuclear_norm = self.svd_enhanced_rank_constraint(proj_Diversity_Matrix_CLS)
                    nuclear_norm = self.svd_enhanced_rank_constraint(Diversity_Matrix)
                    layer_weights = self.generate_descending_weights(num_layers).to(device)
                    log_norm = torch.log(nuclear_norm + 1e-6)
                    beta = 2.5
                    beta_expo = torch.pow(torch.abs(log_norm), beta)
                    FC_loss = -(beta_expo * layer_weights.view(-1, 1)).mean(dim=1).sum()
                    # print(f"fc_loss: {FC_loss}")
                    # 计算FC_loss梯度
                    FC_loss.backward()
                    adv_imgs.retain_grad()
                    fc_grad = adv_imgs.grad.detach().clone()
                    adv_imgs.grad.zero_()
                    # model.zero_grad()
                    # 投影操作
                    sa_flat = sa_grad.flatten()
                    fc_flat = fc_grad.flatten()
                    dot = torch.dot(fc_flat, sa_flat)
                    sa_norm_sq = torch.norm(sa_flat) ** 2 + 1e-6
                    projected_fc = fc_grad - (dot / sa_norm_sq) * sa_grad

                    # 合并梯度（SA梯度主导 + 投影后的FC梯度）
                    final_grad = sa_grad + 0.36 * projected_fc

                # --- 更新对抗样本 ---
                # 梯度归一化
                final_grad = final_grad / torch.mean(torch.abs(final_grad), dim=(1,2,3), keepdim=True)
                # 生成扰动
                perturbation = self.step_size * final_grad.sign()
                # 更新并裁剪
                adv_imgs = torch.clamp(
                    clone_adv_imgs.detach() + perturbation,
                    imgs - self.eps,
                    imgs + self.eps
                ).clamp(0, 1)
                adv_imgs = adv_imgs.detach()
                # 空间管理
                del final_grad, sa_grad, fc_grad, FC_loss, SA_loss, perturbation, sa_flat, fc_flat, dot, sa_norm_sq, projected_fc
                torch.cuda.empty_cache()
                last_adv_imgs = clone_adv_imgs.clone()
            else:
                last_adv_imgs = adv_imgs.clone()
                clear_visual_hook_outputs()
                adv_imgs.requires_grad_()
                scaled_imgs = self.get_scaled_imgs(adv_imgs, [0.5, 0.75, 1.25, 1.5], device)

                # 为最后一张图像获取中间输出    
                if self.normalization is not None:
                    adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))
                    if self.args.source_model in ['ALBEF', 'TCL']:
                        inter_visual_embeddings = []
                        for layer_idx in sorted(visual_self_attn_outputs.keys()):
                            layer_embed = torch.stack(visual_self_attn_outputs[layer_idx], dim=0).squeeze(0)  # [batch, seq, embed]
                            inter_visual_embeddings.append(layer_embed)
                        inter_visual_embeddings = torch.stack(inter_visual_embeddings, dim=0)  # [layers, batch, seq, embed]
                    elif self.args.source_model in ['CLIP_ViT','CLIP_ViT-L/14']:
                        inter_visual_embeddings = []
                        for layer_idx in sorted(visual_self_attn_outputs.keys()):
                            layer_embed = torch.stack(visual_self_attn_outputs[layer_idx], dim=0).squeeze(0).permute(1, 0, 2)  # [batch, seq, embed]
                            inter_visual_embeddings.append(layer_embed)
                        inter_visual_embeddings = torch.stack(inter_visual_embeddings, dim=0)  # [layers, batch, seq, embed]                        
                else:
                    adv_imgs_output = model.inference_image(scaled_imgs)
                    if self.args.source_model in ['ALBEF', 'TCL']:
                        inter_visual_embeddings = []
                        for layer_idx in sorted(visual_self_attn_outputs.keys()):
                            layer_embed = torch.stack(visual_self_attn_outputs[layer_idx], dim=0).squeeze(0)
                            inter_visual_embeddings.append(layer_embed)
                        inter_visual_embeddings = torch.stack(inter_visual_embeddings, dim=0)
                    elif self.args.source_model in ['CLIP_ViT','CLIP_ViT-L/14']:
                        inter_visual_embeddings = []
                        for layer_idx in sorted(visual_self_attn_outputs.keys()):
                            torch.stack(visual_self_attn_outputs[layer_idx], dim=0).squeeze(0).permute(1, 0, 2)
                            inter_visual_embeddings.append(layer_embed)
                        inter_visual_embeddings = torch.stack(inter_visual_embeddings, dim=0)
                adv_imgs_embeds = adv_imgs_output['image_feat']
                model.zero_grad()
                with torch.enable_grad():
                    # --- 计算对抗损失 SA_loss ---
                    SA_loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                    for i in range(5):
                        loss_item = self.loss_func(adv_imgs_embeds[i * b:i * b + b], txt_embeds, txt2img, all_txt_supervisions)
                        SA_loss += loss_item
                    # print(f"SA_loss: {SA_loss}")
                    adv_imgs.retain_grad()  # 确保能获取对抗图像的梯度
                    # 计算SA_loss梯度
                    SA_loss.backward(retain_graph=True)
                    sa_grad = adv_imgs.grad.detach().clone()
                    # 清除之前的梯度
                    adv_imgs.grad.zero_()
                    # 计算FC_Loss
                    # --- 计算正则项 FC_loss ---
                    model.zero_grad()
                    num_layers = inter_visual_embeddings.shape[0] - 1 #减掉最后一层
                    inter_shape_x = inter_visual_embeddings.shape[2]
                    one_matrix = torch.ones((inter_shape_x, 1)).to(device)
                    rank1_approx = inter_visual_embeddings[:-1,:,:,:].sum(dim=2, keepdim=True) / inter_shape_x
                    Diversity_Matrix = inter_visual_embeddings[:-1,:,:,:] - one_matrix @ rank1_approx# 
                    # nuclear_norm = self.svd_enhanced_rank_constraint(Diversity_Matrix,inter_visual_embeddings[-1][:,0,:]) #传入最后一层CLS Token
                    # frobenius_norm = torch.norm(Diversity_Matrix.mean(dim=1), p='fro',dim=(1,2))
                    # Diversity_Matrix_CLS = F.normalize(model.vision_proj(Diversity_Matrix.mean(dim=2,keepdim=True)),dim=-1).cpu().to(device)
                    # proj_Diversity_Matrix_CLS = Diversity_Matrix_CLS @ self.Q_sX # Proj版
                    # nuclear_norm = self.svd_enhanced_rank_constraint(proj_Diversity_Matrix_CLS)
                    nuclear_norm = self.svd_enhanced_rank_constraint(Diversity_Matrix)
                    layer_weights = self.generate_descending_weights(num_layers).to(device)
                    log_norm = torch.log(nuclear_norm + 1e-6)
                    beta = 2.5
                    beta_expo = torch.pow(torch.abs(log_norm), beta)
                    FC_loss = -(beta_expo * layer_weights.view(-1, 1)).mean(dim=1).sum(dim=0)
                    # print(f"fc_loss: {FC_loss}")
                    FC_loss.backward()
                    fc_grad = adv_imgs.grad.detach().clone()
                    adv_imgs.grad.zero_()
                    # 步骤4：梯度投影（直接操作对抗图像梯度）
                    sa_grad_flat = sa_grad.flatten()
                    fc_grad_flat = fc_grad.flatten()
                    dot_product = torch.dot(fc_grad_flat, sa_grad_flat)
                    sa_norm_sq = torch.norm(sa_grad_flat)**2 + 1e-6
                    projected_fc_grad = fc_grad - (dot_product / sa_norm_sq) * sa_grad
                    # 步骤5：合并梯度（对抗梯度主导 + 投影后的正则梯度）
                    final_grad = sa_grad + 0.36 * projected_fc_grad  # 可调节0.217为beta参数
                    # --- 更新对抗图像（快速无优化器版本）---
                    # 梯度归一化
                    final_grad = final_grad / torch.mean(torch.abs(final_grad), dim=(1,2,3), keepdim=True)
                    # 生成扰动（Sign攻击）
                    perturbation = self.step_size * final_grad.sign()
                    # 更新并裁剪对抗样本
                    adv_imgs = torch.clamp(
                        torch.min(
                            torch.max(
                                adv_imgs + perturbation, 
                                imgs - self.eps
                            ), 
                            imgs + self.eps
                        ), 
                        0.0, 1.0
                    )  
                    adv_imgs = adv_imgs.detach()
                    # 空间管理
                    del final_grad, sa_grad, fc_grad, FC_loss, SA_loss, perturbation, sa_grad_flat, fc_grad_flat, dot_product, sa_norm_sq, projected_fc_grad
                    torch.cuda.empty_cache()
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"The function execution time: {elapsed_time:.2f} seconds")
        # 注销 hook，释放内存
        # exit(1)
        for h in visual_hooks:
            h.remove()
        return adv_imgs, last_adv_imgs

    def save_img(self, img_name, norm_img):
        pil_array = (norm_img * 255).to(torch.uint8).cpu().numpy()
        pil_img = Image.fromarray(np.transpose(pil_array, (1, 2, 0)))
        img_path = "./mscoco_imgs/"
        pil_img.save(img_path + img_name)

    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])

        reverse_transform = transforms.Resize(ori_shape,
                                              interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio * ori_shape[0]),
                           int(ratio * ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                                interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)

            reversed_imgs = reverse_transform(scaled_imgs)

            result.append(reversed_imgs)

        return torch.cat([imgs, ] + result, 0)


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)


class TFCTextAttacker():
    def __init__(self, ref_net, ref_net_d, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10,
                 threshold_pred_score=0.3, batch_size=32, text_ratios=[0.6, 0.2, 0.2]):
        self.ref_net = ref_net
        self.ref_net_d = ref_net_d
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.text_ratios = text_ratios

    def img_guided_attack(self, net, texts, img_embeds=None, adv_img_embeds=None, last_adv_img_embeds=None):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_feat'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_feat'].flatten(1)

                if adv_img_embeds == None:
                    loss = self.loss_func(replace_embeds, img_embeds, i)
                else:
                    loss = self.text_ratios[0] * self.loss_func(replace_embeds, img_embeds, i) + self.text_ratios[
                        1] * self.loss_func(replace_embeds, adv_img_embeds, i) + self.text_ratios[2] * self.loss_func(
                        replace_embeds, last_adv_img_embeds, i)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
        loss = loss_TaIcpos
        return loss

    def attack(self, net, texts):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_embed'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='none')
        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_embed'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_embed'].flatten(1)

                loss = criterion(replace_embeds.log_softmax(dim=-1),
                                 origin_embeds[i].softmax(dim=-1).repeat(len(replace_embeds), 1))

                loss = loss.sum(dim=-1)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True,
                                               max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1),
                                  origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words

    # def svd_enhanced_rank_constraint(self, diversity_matrix, final_cls_tokens, compress_ratio=0.7):
    #     """ 全批量处理 + 向量化，效率提升约50倍 """
    #     L, B, S, E = diversity_matrix.shape
    #     device = diversity_matrix.device
        
    #     # 重塑为 [L*B, S, E] 进行批量SVD
    #     reshaped = diversity_matrix.view(L*B, S, E)
    #     U, S_vec, Vh = torch.linalg.svd(reshaped, full_matrices=False)  # S_vec形状 [L*B, K] (K=min(S,E))
    #     K = S_vec.shape[1]
        
    #     # 扩展CLS Token到 [L*B, E]
    #     cls_expanded = final_cls_tokens[None, :, :].expand(L, B, E).reshape(L*B, E)
        
    #     # 批量计算所有右奇异向量与CLS的相关系数 (向量化)
    #     Vh_flat = Vh.reshape(L*B*K, E)                     # [L*B*K, E]
    #     cls_repeated = cls_expanded.repeat_interleave(K, 0) # [L*B*K, E]
        
    #     # 皮尔逊相关系数 (向量化)
    #     cov = torch.einsum('ne,ne->n', Vh_flat - Vh_flat.mean(dim=1, keepdim=True), 
    #                     (cls_repeated - cls_repeated.mean(dim=1, keepdim=True)))
    #     std_v = torch.std(Vh_flat, dim=1)                   # [L*B*K]
    #     std_cls = torch.std(cls_repeated, dim=1)             # [L*B*K]
    #     corr_coeffs = (cov / (std_v * std_cls + 1e-8)).abs().view(L*B, K)  # [L*B, K]
        
    #     # 动态计算保留成分 (全批量操作)
    #     keep_counts = max(1, int(K * (1 - compress_ratio)))
    #     weighted_scores = corr_coeffs * (S_vec / (S_vec.sum(dim=1, keepdim=True) + 1e-8))
    #     _, topk_indices = torch.topk(weighted_scores, k=keep_counts, dim=1)  # [L*B, keep_counts]
        
    #     # 批量生成mask
    #     mask = torch.zeros_like(S_vec, dtype=torch.bool)
    #     mask.scatter_(1, topk_indices, True)
        
    #     # 核范数损失计算 (保留部分0.1，压缩部分1.0)
    #     loss_retain = (S_vec * mask).sum(dim=1) * 0.1       # [L*B]
    #     loss_compress = (S_vec * ~mask).sum(dim=1) * 1.0     # [L*B]
    #     nuclear_loss = (loss_retain + loss_compress).view(L, B)
        
    #     return nuclear_loss

    # def svd_enhanced_rank_constraint(self, diversity_matrix, final_cls_tokens, compress_ratio=0.7):
    #     L, B, S, E = diversity_matrix.shape
    #     device = diversity_matrix.device
        
    #     # 重塑为 [L*B, S, E] 进行批量SVD
    #     reshaped = diversity_matrix.view(L*B, S, E)
    #     U, S_vec, Vh = torch.linalg.svd(reshaped, full_matrices=False)  # S_vec形状 [L*B, K] (K=min(S,E))
    #     K = S_vec.shape[1]
        
    #     # 扩展CLS Token到 [L*B, E]
    #     cls_expanded = final_cls_tokens[None, :, :].expand(L, B, E).reshape(L*B, E)
        
    #     # 批量计算所有右奇异向量与CLS的Spearman系数
    #     Vh_flat = Vh.reshape(L*B*K, E)                     # [L*B*K, E]
    #     cls_repeated = cls_expanded.repeat_interleave(K, 0) # [L*B*K, E]
        
    #     # 计算Spearman系数（替换原皮尔逊部分）
    #     def compute_spearman(x, y):
    #         x_rank = torch.argsort(torch.argsort(x, dim=1), dim=1).float()
    #         y_rank = torch.argsort(torch.argsort(y, dim=1), dim=1).float()
    #         x_centered = x_rank - x_rank.mean(dim=1, keepdim=True)
    #         y_centered = y_rank - y_rank.mean(dim=1, keepdim=True)
    #         cov = (x_centered * y_centered).sum(dim=1)
    #         x_std = torch.std(x_rank, dim=1)
    #         y_std = torch.std(y_rank, dim=1)
    #         corr = cov / (x_std * y_std + 1e-8)
    #         return corr.abs()
        
    #     corr_coeffs = compute_spearman(Vh_flat, cls_repeated).view(L*B, K)
        
    #     # 动态计算保留成分
    #     keep_counts = max(1, int(K * (1 - compress_ratio)))
    #     weighted_scores = corr_coeffs * (S_vec / (S_vec.sum(dim=1, keepdim=True) + 1e-8))
    #     _, topk_indices = torch.topk(weighted_scores, k=keep_counts, dim=1)
        
    #     # 批量生成mask
    #     mask = torch.zeros_like(S_vec, dtype=torch.bool)
    #     mask.scatter_(1, topk_indices, True)
        
    #     # 核范数损失计算
    #     loss_retain = (S_vec * mask).sum(dim=1)
    #     loss_compress = (S_vec * ~mask).sum(dim=1)
    #     nuclear_loss =  (loss_retain+loss_compress).view(L, B)
        
    #     return nuclear_loss

    # def svd_enhanced_rank_constraint(self, matrix):
    #     U, S_vec, Vh = torch.linalg.svd(matrix)
    #     energy_loss = torch.sum(S_vec**2)  # 抑制总能量
    #     entropy_loss = -torch.sum(torch.log(S_vec + 1e-6))  # 避免过度稀疏化
    #     return energy_loss + 0.1 * entropy_loss

    # def svd_enhanced_rank_constraint(self, vis_matrix, text_embeddings, k_ratio=0.3, compress_topk=True):
    #     """
    #     vis_matrix: [Layer_image, 5*N, 577, 768]
    #     text_embeddings: [Layer_text, 5*N, 30, 768]
    #     """
    #     L_img, B_img, S_v, d = vis_matrix.shape
    #     L_txt, B_txt, S_t, _ = text_embeddings.shape
    #     N = B_img // 5
        
    #     # =====================
    #     # 1. 输入重组（确保连续性）
    #     # =====================
    #     # 文本特征加权融合
    #     layer_weights_text = torch.linspace(1.0, 0.217, steps=L_txt).to(text_embeddings.device)
    #     text_embeddings_weighted = (text_embeddings * layer_weights_text.view(-1, 1, 1, 1)).sum(dim=0)  # [5*N, 30, d]
    #     # 图像和文本组重组
    #     vis_groups = vis_matrix.reshape(L_img, N, 5, S_v, d)          # [L_img, N, 5, 577, d]
    #     txt_groups = text_embeddings_weighted.reshape(N, 5, S_t, d)   # [N, 5, 30, d]
    #     txt_fused = txt_groups.mean(dim=2)                            # [N, 5, d]
    #     # =====================
    #     # 2. 组内独立处理
    #     # =====================
    #     nuclear_loss_list = []
    #     for n in range(N):
    #         vis_group = vis_groups[:, n, :, :, :].contiguous()  # [L_img, 5, 577, d]
    #         txt_group = txt_fused[n, :, :]                       # [5, d]
            
    #         # 合并Layer和增强维度
    #         L_flat = L_img * 5
    #         vis_flat = vis_group.reshape(L_flat, S_v, d)
            
    #         # 通道-文本相似度计算
    #         vis_ch_mean = vis_flat.mean(dim=1)             # [L_flat, d]
    #         txt_expanded = txt_group.repeat(L_img, 1)      # [L_flat, d]

    #         sim_matrix = F.cosine_similarity(
    #             vis_ch_mean.transpose(0, 1), 
    #             txt_expanded.transpose(0, 1)
    #         )
            
    #         # 生成通道掩码
    #         k = int(d * k_ratio)
    #         if compress_topk:
    #             _, topk_indices = torch.topk(sim_matrix, k, dim=-1, largest=True)
    #         else:
    #             _, topk_indices = torch.topk(sim_matrix, k, dim=-1, largest=False)
    #         mask = torch.ones_like(sim_matrix, dtype=torch.bool)
    #         mask.scatter_(-1, topk_indices, False)
            
    #         # 分割特征矩阵
    #         vis_kept = vis_flat[:, :, ~mask].contiguous()  # 高相关通道 [L_flat, S_v, k]
    #         vis_rest = vis_flat[:, :, mask].contiguous()   # 低相关通道 [L_flat, S_v, d-k]

    #         # 双通道SVD处理
    #         def compute_svd_loss(features):
    #             U, S_vec, _ = torch.linalg.svd(features, full_matrices=False)
    #             return S_vec.sum(dim=-1)  # 沿奇异值维度求和 [L_flat]

    #         # 计算两个子矩阵的核范数损失
    #         loss_kept = compute_svd_loss(vis_kept)
    #         loss_rest = compute_svd_loss(vis_rest)
            
    #         # 加权融合损失
    #         combined_loss = loss_kept + 0.25 * loss_rest
            
    #         # 保持形状一致性
    #         nuclear_loss_list.append(combined_loss.view(L_img, 5))
        
    #     # =====================
    #     # 3. 重组输出
    #     # =====================
    #     nuclear_loss = torch.cat(nuclear_loss_list, dim=1)  # [Layer_image, 5*N]
    #     return nuclear_loss

    # def svd_enhanced_rank_constraint_one(self, vis_matrix, text_embeddings, k_ratio=0.3, compress_topk=True):
    #     """
    #     vis_matrix: [Layer_image, N, 577, 768]
    #     text_embeddings: [Layer_text, 5*N, 30, 768]
    #     """
    #     L_img, B_img, S_v, d = vis_matrix.shape
    #     L_txt, B_txt, S_t, _ = text_embeddings.shape
    #     N = B_img  # 直接使用输入维度
        
    #     # =====================
    #     # 1. 输入重组（文本部分保持原处理）
    #     # =====================
    #     # 文本特征加权融合
    #     layer_weights_text = torch.linspace(1.0, 0.217, steps=L_txt).to(text_embeddings.device)
    #     text_embeddings_weighted = (text_embeddings * layer_weights_text.view(-1, 1, 1, 1)).sum(dim=0)  # [5*N, 30, d]
        
    #     # 文本组重组
    #     txt_groups = text_embeddings_weighted.reshape(N, 5, S_t, d)   # [N, 5, 30, d]
    #     txt_fused = txt_groups.mean(dim=2)                            # [N, 5, d]
        
    #     nuclear_loss_list = []
    #     for n in range(N):
    #         # =====================
    #         # 2. 图像特征扩展处理
    #         # =====================
    #         # 获取图像特征并扩展5次
    #         vis_group = vis_matrix[:, n, :, :]  # [L_img, 577, d]
    #         vis_group = vis_group.unsqueeze(1).expand(-1, 5, -1, -1)  # [L_img,5,577,d]
            
    #         # 合并Layer和增强维度
    #         L_flat = L_img * 5
    #         vis_flat = vis_group.reshape(L_flat, S_v, d)  # [L_flat,577,768]
            
    #         # =====================
    #         # 3. 相似度计算（修正维度处理）
    #         # =====================
    #         # 通道特征均值
    #         vis_ch_mean = vis_flat.mean(dim=1)  # [L_flat, d]
            
    #         # 文本特征扩展
    #         txt_group = txt_fused[n, :, :]    # [5, d]
    #         txt_expanded = txt_group.repeat(L_img, 1)  # [L_flat, d]
            
    #         # 计算通道重要性
    #         sim_matrix = F.cosine_similarity(
    #             vis_ch_mean.transpose(0, 1),  # [d, L_flat]
    #             txt_expanded.transpose(0, 1) # [d, L_flat]
    #         )  # 结果形状 [d]
            
    #         # 生成通道掩码
    #         k = int(d * k_ratio)
    #         if compress_topk:
    #             _, topk_indices = torch.topk(sim_matrix, k, dim=-1, largest=True)
    #         else:
    #             _, topk_indices = torch.topk(sim_matrix, k, dim=-1, largest=False)
    #         mask = torch.ones_like(sim_matrix, dtype=torch.bool)
    #         mask.scatter_(-1, topk_indices, False)
            
    #         # 分割特征矩阵
    #         vis_kept = vis_flat[:, :, ~mask].contiguous()  # 高相关通道 [L_flat, S_v, k]
    #         vis_rest = vis_flat[:, :, mask].contiguous()   # 低相关通道 [L_flat, S_v, d-k]

    #         # SVD计算函数
    #         def compute_svd_loss(features):
    #             U, S_vec, _ = torch.linalg.svd(features, full_matrices=False)
    #             return S_vec.sum(dim=-1) # 按原层结构重组

    #         # 计算并融合损失
    #         loss_kept = compute_svd_loss(vis_kept)
    #         loss_rest = compute_svd_loss(vis_rest)
    #         combined_loss = loss_kept + 0.25 * loss_rest
            
    #         # 按层取平均
    #         nuclear_loss_list.append(combined_loss.view(L_img, 5))  # [L_img]
        
    #     # =====================
    #     # 5. 重组输出
    #     # =====================
    #     nuclear_loss = torch.cat(nuclear_loss_list, dim=1)  # [Layer_image, N]
    #     return nuclear_loss
    # SVD增强的秩约束版本
    # def svd_enhanced_rank_constraint(self, matrix, k=2):
    #     """ 对每个layer-batch组合的[Seq, Embed]矩阵进行SVD约束，保留前k个奇异值，压缩剩余部分 """
    #     # 合并layer和batch维度
    #     L, B, S, E = matrix.shape
    #     reshaped = matrix.view(L*B, S, E)
        
    #     # 批处理SVD
    #     U, S_vec, Vh = torch.linalg.svd(reshaped, full_matrices=False)
    #     # print(S_vec.shape)
        
    #     # 核范数计算（仅压缩后k个奇异值）
    #     nuclear_loss = S_vec[:, :].mean(dim=-1) - S_vec[:, S_vec.shape[-1]//2] # 形状 [L*B]
        
    #     return nuclear_loss.view(L, B)  # 恢复为 [layer, batch]

    # def svd_enhanced_rank_constraint(self, matrix):
    #     """ 计算奇异值的均匀性惩罚项（方差或熵） """
    #     L, B, S, E = matrix.shape
    #     reshaped = matrix.view(L*B, S, E)
        
    #     # 批处理SVD
    #     U, S_vec, Vh = torch.linalg.svd(reshaped, full_matrices=False)
        
    #     # --- 均匀性惩罚计算 ---
    #     # 归一化奇异值为概率分布（避免除以零）
    #     sigma_sum = torch.sum(S_vec, dim=1, keepdim=True)  # [L*B, 1]
    #     p = S_vec / (sigma_sum + 1e-6)  # [L*B, min(S,E)]
        
    #     # 方法1：计算方差（方差越小越均匀）
    #     # mean_p = torch.mean(p, dim=1, keepdim=True)
    #     # var_loss = torch.mean((p - mean_p)**2, dim=1)  # [L*B]
        
    #     # 方法2（可选）：计算熵（熵越大越均匀）
    #     entropy_loss = -torch.sum(p * torch.log(p + 1e-6), dim=1)  # [L*B]
        
    #     return entropy_loss.view(L, B)  # 返回方差损失 [layer, batch]