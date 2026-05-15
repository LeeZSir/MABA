# Third-Party Notices

This repository includes or interfaces with third-party code and model ecosystems. The project-level license is not declared yet; third-party components keep their original licenses and notices.

Before publishing the repository, add a root `LICENSE` file for the MABA code and keep this notice file in sync with any copied code or model-specific evaluation scripts.

## Vendored Code

| Path | Upstream | License / notice |
| --- | --- | --- |
| `models/blip_model/`, `models/base_model.py`, `models/model_*.py` | Salesforce BLIP / LAVIS | BSD-3-Clause. Several files already contain Salesforce copyright and SPDX headers. |
| `models/ulip_models/` | Salesforce ULIP / ULIP-2 | BSD-3-Clause. Files contain Salesforce copyright and SPDX headers; checkpoint weights are not distributed here. |
| `models/clip_model/` | OpenAI CLIP | MIT License. Keep the OpenAI copyright and MIT license notice when redistributing copied CLIP code. |
| `models/tokenization_bert.py`, `models/xbert.py` | BERT / HuggingFace Transformers-derived code | Apache-2.0 headers are present in the files. |

## External Models and Evaluation Backends

The evaluation scripts call external model packages or model weights that are not distributed in this repository. Check each model card and upstream repository before use, especially if redistributing checkpoints or running commercial evaluation.

| Script family | Typical upstream | License notes |
| --- | --- | --- |
| BLIP-2 / LAVIS | Salesforce LAVIS | BSD-3-Clause for the LAVIS codebase. |
| MiniGPT-4 | Vision-CAIR MiniGPT-4 | BSD-3-Clause for the upstream project. |
| LLaVA-OneVision | LLaVA-NeXT | Apache-2.0 for the upstream codebase. |
| Qwen2.5-VL | Qwen model cards / QwenLM repositories | License may vary by model size and release. `Qwen/Qwen2.5-VL-7B-Instruct` is marked `apache-2.0` on Hugging Face at the time of this check. |
| MiniCPM / MiniCPM-V / MiniCPM-o | OpenBMB MiniCPM | Apache-2.0 for the upstream repository and listed MiniCPM models. |
| InternVL | OpenGVLab InternVL | Check the exact model card and repository license for the model variant being evaluated. |

## Data

COCO, VQAv2, OK-VQA, TextVQA, VizWiz, and similar benchmark images or annotations are not distributed in this repository except for small local annotation examples. Follow the dataset licenses and terms from the original providers.

