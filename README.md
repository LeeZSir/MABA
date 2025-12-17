# MABA

**Paper:** *On Evaluating the Robustness of Large Vision-Language Models via Untargeted Modality Alignment Breaking Adversarial Attack*

This repository contains the official implementation of **MABA**, released for the anonymous review of our **USENIX Security** submission.  
License and authorship information will be updated upon acceptance.

---

## 🚩 Overview

We propose an **Untargeted Modality Alignment Breaking Adversarial Attack (MABA)** to evaluate the robustness of Large Vision-Language Models (LVLMs).

This repository includes:

- Training scripts for the **MIA Projector**
- Adversarial attack generation scripts
- Evaluation scripts for:
  - Image Captioning (IC)
  - Visual Question Answering (VQA)
- Illustrative adversarial example images

The evaluation pipeline is primarily adapted from **Qwen-VL**.

---

## ⚙️ Environment Preparation

### 1. Basic Installation

```bash
pip install -r requirements.txt
**
### 2. Model Configuration

To test a wide range of LVLM families, we **strongly recommend** creating **separate virtual environments** for different models, as they often rely on incompatible `transformers` versions.

| Model Family     | Official Repository                                                                                                                                |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| BLIP-2           | [https://github.com/salesforce/LAVIS/tree/main/projects/blip2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)                       |
| MiniGPT-4        | [https://github.com/Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)                                                               |
| LLaVA-OneVision  | [https://github.com/LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)                                                                   |
| Phi-4-Multimodal | [https://github.com/marketplace/models/azureml/Phi-4-multimodal-instruct](https://github.com/marketplace/models/azureml/Phi-4-multimodal-instruct) |
| Qwen-VL          | [https://github.com/QwenLM/Qwen-VL](https://github.com/QwenLM/Qwen-VL)                                                                             |
| MiniCPM          | [https://github.com/OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM)                                                                           |
| InternVL3        | [https://github.com/OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL)                                                                     |



## 🚀 Usage

### 1. Train the MIA Projector

```bash
python mSMINE.py \
  --cuda_id 0 \
  --source_model CLIP_ViT-L/14 \
  --batch_size 10
```

---

### 2. Configure the Attack

#### A. Set Projector Path (Line 63)

```python
self.checkpoint_X = torch.load(
    'YOUR_SLICING_MATRIX.pt',
    map_location='cpu'
)
```

---

#### B. Set Beta Parameter (Lines 313 & 407)

```python
beta = 2.5
```

---

#### C. Select Target Layers (Lines 168 & 176)

```python
visual_self_attn_outputs = {i: [] for i in list(range(11, 17)) + [23]}

for i in list(range(11, 17)) + [23]:
```

---

### 3. Generate Adversarial Examples

```bash
python test_attack.py \
  --cuda_id 0 \
  --batch_size 10 \
  --source_model CLIP_ViT-L/14
```

---

### 4. Evaluation

#### Standard Metrics

* `eval_IC/`
* `eval_vqa/`

#### LLM-based Evaluation

```bash
python llm_judge.py
```

---

## 💐 Acknowledgements

* [https://github.com/jiaxiaojunQAQ/SA-AET](https://github.com/jiaxiaojunQAQ/SA-AET)
* [https://github.com/QwenLM/Qwen-VL](https://github.com/QwenLM/Qwen-VL)

---
