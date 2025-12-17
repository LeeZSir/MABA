# MABA

Paper code for *"On Evaluating the Robustness of Large Vision-Language Models via Untargeted Modality Alignment Breaking Adversarial Attack"*.

This code is released for the anonymous review of our USENIX submission. License and authorship information will be updated upon acceptance.

We also include evaluation scripts and several illustrative adversarial example images for reference. The Image Captioning (IC) and Visual Question Answering (VQA) evaluation code is mainly adapted from [Qwen-VL].

## Environment Preparation & Model Configuration

First, install the basic environment required for the attack:

```bash
pip install -r requirements.txt

To test a wider range of model families, we strongly recommend configuring separate virtual environments for different models, as these model libraries are often sensitive to specific transformers versions.
This study evaluates the robustness of the following Large Vision-Language Models (LVLMs). Please refer to the official repositories listed below for installation guidelines and model weight downloads:

- **BLIP-2**: [https://github.com/salesforce/LAVIS/tree/main/projects/blip2]
- **MiniGPT-4**: [https://github.com/Vision-CAIR/MiniGPT-4]
- **LLaVA-OneVision**: [https://github.com/LLaVA-VL/LLaVA-NeXT]
- **Phi-4-Multimodal**: [https://github.com/marketplace/models/azureml/Phi-4-multimodal-instruct]
- **Qwen-VL**: [https://github.com/QwenLM/Qwen-VL]
- **MiniCPM**: [https://github.com/OpenBMB/MiniCPM]
- **InternVL3**: [https://github.com/OpenGVLab/InternVL]

## Usage

1.  **Train the MIA Projector**  
    Execute `mSMINE.py` to train the Modality Interaction Alignment (MIA) projector as described in the paper.
    ```bash
    python mSMINE.py --cuda_id 0 --source_model CLIP_ViT-L/14 --batch_size 10
    ```

2.  **Configure the Attack**  
    Update the configuration in `MABA.py` to point to your trained projector and set the attack hyperparameters:

    *   **Set Projector Path (Line 63):** Specify the location of the generated MIA projector checkpoint.
        ```python
        # Load mSMI Matrix
        self.checkpoint_X = torch.load('YOUR_SLICING_MATRIX.pt', map_location='cpu')
        ```
    *   **Set Beta Parameter (Lines 313 & 407):** Configure the $\beta$ hyperparameter.
        ```python
        beta = 2.5
        ```
    *   **Select Target Layers (Lines 168 & 176):** Define the specific layers used for the attack.
        ```python
        visual_self_attn_outputs = {i: [] for i in list(range(11,17)) + [23]}
        # ...
        for i in list(range(11,17)) + [23]:
        ```

3.  **Generate Adversarial Examples**  
    Run `test_attack.py` to execute the attack and generate adversarial images.
    ```bash
    python test_attack.py --cuda_id 0 --batch_size 10 --source_model CLIP_ViT-L/14
    ```

4.  **Evaluation**  
    *   **Standard Metrics:** Use the scripts located in the `eval_IC` (Image Captioning) and `eval_vqa` (Visual Question Answering) directories to generate model responses and compute standard performance metrics.
    *   **LLM-based Evaluation:** Execute `llm_judge.py` to perform the LLM-as-a-Judge assessment.

## Acknowledgements
This codebase is partially based on the implementation from:
- [jiaxiaojunQAQ/SA-AET]
- [QwenLM/Qwen-VL]
