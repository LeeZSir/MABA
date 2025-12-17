# MABA

> **Paper:** *On Evaluating the Robustness of Large Vision-Language Models via Untargeted Modality Alignment Breaking Adversarial Attack*

This repository contains the official implementation code for the paper **MABA**. This code is released for the anonymous review of our **USENIX** submission.

License and authorship information will be updated upon acceptance.

---

## 🚩 Overview

We propose an Untargeted Modality Alignment Breaking Adversarial Attack (MABA) to evaluate the robustness of LVLMs. This repository includes:
*   Training scripts for the MIA Projector.
*   Attack generation scripts.
*   Evaluation scripts for Image Captioning (IC) and Visual Question Answering (VQA).
*   Illustrative adversarial example images.

The evaluation code is primarily adapted from [Qwen-VL].

## ⚙️ Environment Preparation

### 1. Basic Installation
First, install the basic environment requirements:

```bash
pip install -r requirements.txt
