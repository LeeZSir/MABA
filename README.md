# MABA

Official implementation of **MABA**, an untargeted modality alignment breaking adversarial attack for evaluating the robustness of Large Vision-Language Models.

Paper: *On Evaluating the Robustness of Large Vision-Language Models via Untargeted Modality Alignment Breaking Adversarial Attack*

## Repository Layout

- `mSMINE.py`: trains the MIA projector / slicing matrix.
- `test_attack.py`: generates adversarial images with MABA.
- `MABA.py`: image and text attacker implementations.
- `eval_IC/`: image captioning evaluation scripts.
- `eval_vqa/`: VQA evaluation scripts.
- `llm_judge.py`: LLM-based caption and VQA judging.
- `configs/Retrieval_coco.yaml`: minimal COCO-style example config.
- `data_annotation/`: small annotation files used by examples.
- `adv_image/`: illustrative adversarial examples.

## Environment

Install the base dependencies used by the attack pipeline:

```bash
pip install -r requirements.txt
```

The evaluation scripts target several LVLM families whose dependencies can conflict, especially around `transformers`, CUDA kernels, and model-specific packages. Use separate virtual environments for BLIP-2/LAVIS, MiniGPT-4, LLaVA-OneVision, Phi-4, Qwen-VL, MiniCPM, and InternVL. This repository intentionally does not provide one unified environment file for all evaluators.

## Data and Checkpoints

The attack and projector scripts read a small YAML config:

```yaml
test_file: ./data_annotation/coco_test_sub.json
image_root: ./datasets_own/MSCOCO
embed_dim: 256
num_epochs: 100
```

Set `image_root` to your local COCO image directory. Checkpoints, generated images, logs, and evaluation outputs are not committed and are ignored by `.gitignore`.

## Train the Slicing Matrix

```bash
python mSMINE.py \
  --config ./configs/Retrieval_coco.yaml \
  --cuda_id 0 \
  --source_model CLIP_ViT-L/14 \
  --source_text_encoder bert-base-uncased \
  --batch_size 10 \
  --output_dir ./slicing_matrix/checkpoints \
  --log qformer_l14
```

The checkpoint is written under:

```text
./slicing_matrix/checkpoints/<timestamp>/<log>/slicing_matrices_epoch_<N>.pt
```

## Generate Adversarial Images

Pass the slicing matrix explicitly:

```bash
python test_attack.py \
  --config ./configs/Retrieval_coco.yaml \
  --cuda_id 0 \
  --batch_size 10 \
  --source_model CLIP_ViT-L/14 \
  --source_text_encoder bert-base-uncased \
  --slicing_matrix_path ./slicing_matrix/checkpoints/<timestamp>/qformer_l14/slicing_matrices_epoch_100.pt \
  --output_dir ./adv_output \
  --log MABA
```

Or use an environment variable:

```bash
export MABA_SLICING_MATRIX=/path/to/slicing_matrices_epoch_100.pt
python test_attack.py --config ./configs/Retrieval_coco.yaml
```

Generated images are saved to:

```text
./adv_output/<log>/val2014/
```

## Evaluation

Each evaluator has a lightweight `--help` path that does not import model-specific packages. Inspect the options before running a model family:

```bash
python eval_IC/eval_QwenVL_IC.py --help
python eval_vqa/eval_vqa_QwenVL2.5.py --help
```

Common arguments include:

- `--data_path` / `--dataset`: annotation source.
- `--image_path` / `--image_root`: clean or adversarial image root.
- `--output_dir` / `--out_dir`: result directory.
- `--model_path`: Hugging Face model id or local model directory.
- `--cuda_visible_devices`: GPU selection for scripts that set CUDA visibility.

Example VQA run after installing the Qwen2.5-VL environment:

```bash
python eval_vqa/eval_vqa_QwenVL2.5.py \
  --dataset vqav2_val \
  --image_root ./adv_output/MABA/val2014 \
  --out_dir ./vqa_outputs \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct
```

## LLM Judge

API keys are read only from environment variables; they are not set in code.

```bash
export OPENAI_API_KEY=...
python llm_judge.py \
  --ref ./data_annotation/coco_test_sub.json \
  --gen ./caption_outputs/MABA/adv.json \
  --model gpt-4o
```

Other providers use their standard environment variables, including `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, `GENAI_API_KEY`, and `GROQ_API_KEY`.

## Reproducibility Notes

- The default text encoder is `bert-base-uncased`. Use `--source_text_encoder` for a local or alternative checkpoint.
- The slicing matrix must be supplied through `--slicing_matrix_path` or `MABA_SLICING_MATRIX`; missing paths fail fast.
- Path configuration, import cleanup, and README updates do not change MABA or mSMINE algorithm choices such as beta, layer selection, gradient combination, or loss definitions.
- If you change core attack or projector behavior, document it as an algorithm change.

## License and Third-Party Code

This repository does not yet declare a root project license. Add a root `LICENSE` before public release, because several vendored files refer to a repo-root license file.

See `THIRD_PARTY_NOTICES.md` for copied or adapted third-party code and model ecosystem notes. In particular, BLIP/LAVIS/ULIP code carries Salesforce BSD-3-Clause notices, CLIP code is MIT licensed, and BERT/Transformers-derived files carry Apache-2.0 notices.

## Acknowledgements

- https://github.com/jiaxiaojunQAQ/SA-AET
- https://github.com/QwenLM/Qwen-VL
- https://github.com/openai/CLIP
- https://github.com/salesforce/LAVIS
- https://github.com/salesforce/ULIP
