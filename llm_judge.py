import base64
import json
import logging
import os
import re
import sys
import tempfile
import time
from http import HTTPStatus
from io import BytesIO
from typing import Dict, List
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate caption or VQA outputs with an LLM judge.")
    parser.add_argument("--ref", type=str, default="./data_annotation/coco_test_sub.json")
    parser.add_argument("--gen", type=str, help="Path to generated captions JSON")
    parser.add_argument("--img_path", type=str, help="Image root for multimodal caption evaluation")
    parser.add_argument("--inspect", action="store_true", help="Only inspect message structure")
    parser.add_argument("--multimodal", action="store_true", help="Multimodal input")
    parser.add_argument("--vqa", action="store_true", help="VQA")
    parser.add_argument("--ref_answer", type=str, default="./data_annotation/vqav2/v2_mscoco_val2014_annotations_subset.json")
    parser.add_argument("--ref_question", type=str, default="./data_annotation/vqav2/v2_OpenEnded_mscoco_val2014_questions_subset.json")
    parser.add_argument("--gen_answer", type=str, help="Path to generated VQA answers JSON")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM judge model name")
    return parser


if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    build_parser().print_help()
    raise SystemExit(0)

import backoff
import dashscope
import google.generativeai as genai
import openai
import requests
from PIL import Image
from google.api_core.exceptions import InvalidArgument, ResourceExhausted, InternalServerError, BadRequest
from groq import Groq
from requests.exceptions import SSLError

logger = logging.getLogger("desktopenv.agent")

pure_text_settings = ['a11y_tree']

attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"
# More namespaces defined in OSWorld, please check desktop_env/server/main.py


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Please set the {name} environment variable")
    return value


def encoded_img_to_pil_img(image_url: str) -> Image.Image:
    if image_url.startswith("data:image/"):
        _, encoded = image_url.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    if image_url.startswith("file://"):
        image_url = image_url[len("file://"):]
    image_path = os.path.expanduser(image_url)
    if os.path.isfile(image_path):
        return Image.open(image_path).convert("RGB")
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def save_to_tmp_img_file(image_url: str) -> str:
    image = encoded_img_to_pil_img(image_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
        image.save(f, format="PNG")
        return f.name


class PromptAgent:
    def __init__(
        self,
        platform="ubuntu",
        model="gpt-4-vision-preview",
        max_tokens=1500,
        top_p=0.9,
        temperature=0.5
    ):
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.system_message = (
            """You are an expert image caption evaluator. Given a human-written reference and a machine-generated caption, score their semantic similarity from 0 (completely different) to 1 (identical in meaning).
            Criteria:
            - Main Objects (40%): Are the key objects from the reference present? Matching any one reference sentence is enough. Extra objects are fine.
            - Actions & Relationships (30%): Are main actions and relations preserved? Focus on who does what to whom.
            - Attributes (20%): Are core attributes (e.g., color, size, number) correct?
            - Fluency (10%): Is the sentence clear and grammatical? Minor errors are acceptable.
            Adjustments:
            - Add up to +0.05 for especially clear or faithful meaning.
            - Deduct up to −0.1 for serious errors or irrelevant content.
            Final score = sum of parts, clamped to [0, 1]. Reply only with a number rounded to two decimals."""
        )
        # For VQA Evaluation
        # self.system_message = (
        #     "You are a VQA evaluation expert.\n"
        #     "Given the following format:\n"
        #     "Question: {{question}}\n"
        #     "Generated Answer: {{generated answer}}\n"
        #     "Human Answers: {{list of 10 answers}}\n\n"
        #     "Compute the VQA accuracy score using the official formula: min(n / 3, 1), "
        #     "where n is the number of human answers semantically matching the generated answer.\n"
        #     "Return only the vqa score as a single number between 0 and 1, with no explanation or text."
        # )
                # For multimodal Evaluation
        # self.system_message = (
        #         """ You are an expert in image-caption alignment. Given reference captions and an image, score how well the visual content matches the captions on a scale from 0 (unrelated) to 1 (perfectly aligned).
        #             Scoring Criteria:
        #             - Entity Grounding (40%): Do the main objects in the image match those mentioned in the captions? Penalize clearly irrelevant or inconsistent objects.
        #             - Action & Interaction Grounding (30%): Do the actions or interactions shown match those described? Penalize obvious mismatches.
        #             - Attribute Consistency (20%): Are visual attributes (e.g., color, size, count) consistent with the captions? Penalize clear errors.
        #             - Caption Relevance (10%): Is the overall image content relevant to the captions? Penalize unrelated or implausible content.
        #             Final score = sum of parts, clamped to [0, 1]. Reply only with a number rounded to two decimals.
        #         """
        # )

    def clean_caption(self, caption: str) -> str:
        """清洗caption文本以节省tokens"""
        caption = caption.strip()
        caption = re.sub(r"\s+", " ", caption)                      # 合并多空格
        caption = re.sub(r"[\"“”‘’*#<>]", "", caption)              # 去除引号和特殊符号
        caption = re.sub(r"[!?.]{2,}", ".", caption)                # 过多标点缩减
        caption = re.sub(r"\s+([.!?,])", r"\1", caption)            # 去除标点前的空格
        return caption.strip()

    def load_captions(self, reference_file: str, generated_file: str):
        with open(reference_file, 'r') as f:
            reference_data = json.load(f)
        with open(generated_file, 'r') as f:
            generated_data = json.load(f)

        # image_id -> concatenated cleaned reference captions
        ref_dict = {}
        for obj in reference_data:
            image_id = int(obj["image"].split("_")[-1].split(".")[0])
            cleaned_captions = [self.clean_caption(c) for c in obj["caption"]]
            concatenated = " ".join(cleaned_captions).strip()
            ref_dict[image_id] = concatenated

        all_messages = []
        for obj in generated_data:
            image_id = obj["image_id"]
            generated_caption = self.clean_caption(obj["caption"])
            reference_caption = ref_dict.get(image_id)

            if reference_caption:
                messages = [
                    {
                        "role": "system",
                        "content": self.system_message
                    },
                    {
                        "role": "user",
                        "content": (
                            "Rate the semantic similarity between the following two texts on a scale from 0 to 1.\n\n"
                            f"**Reference Captions:** {reference_caption}\n"
                            f"**Generated Caption:** {generated_caption}"
                        )
                    }
                ]
                all_messages.append((image_id, reference_caption, generated_caption, messages))
            else:
                logger.warning(f"No reference captions found for image_id {image_id}")

        return all_messages

    import base64
    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def check_and_print_gpt4o_messages_format(self, all_messages):
        def is_base64_image(data_url: str):
            return isinstance(data_url, str) and data_url.startswith("data:image/") and "base64," in data_url

        for idx, (image_id, caption, messages) in enumerate(all_messages):
            print(f"\n--- Checking Message #{idx+1} (Image ID: {image_id}) ---")
            print(f"📝 Caption: {caption!r}")
            if not isinstance(messages, list):
                print("❌ Error: `messages` should be a list.")
                continue

            for i, msg in enumerate(messages):
                role = msg.get("role")
                content = msg.get("content")
                image_data = msg.get("image", None)

                # Role check
                if role not in {"system", "user", "assistant"}:
                    print(f"❌ Error in message[{i}]: invalid role '{role}'.")
                else:
                    print(f"✅ message[{i}] role: '{role}'")

                # Content check
                if not isinstance(content, str):
                    print(f"❌ Error in message[{i}]: content must be a string.")
                else:
                    print(f"✅ message[{i}] content preview: {content[:60]!r}...")

                # Image check (only if exists)
                if image_data is not None:
                    if is_base64_image(image_data):
                        print(f"✅ message[{i}] includes valid base64 image. Preview: {image_data[:80]!r}...")
                    else:
                        print(f"❌ Error in message[{i}]: invalid base64 image format.")
        
        print("\n✔️ All messages checked.")

    def predict(self, origin_caption_file: str, adv_caption_file: str) -> List[Dict]:
        """
        Perform batch evaluation of caption similarity and save results to JSON.
        """
        message_data = self.load_captions(origin_caption_file, adv_caption_file)

        results = []
        for image_id, ref_caption, gen_caption, messages in message_data:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            }

            try:
                response = self.call_llm(payload)
                logger.info("RESPONSE: %s", response)
                score = float(response.strip())
            except Exception as e:
                logger.error("Failed to call LLM: %s", str(e))
                score = None
            except ValueError:
                logger.warning("Invalid score returned: %s", response)
                score = None

            results.append({
                "image_id": image_id,
                "ref_caption": ref_caption,
                "gen_caption": gen_caption,
                "score": score
            })

        valid_scores = [r["score"] for r in results if r["score"] is not None]
        average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        output = {
            "average_score": average_score,
            "results": results
        }

        with open("caption_eval_results.json", "w") as f:
            json.dump(output, f, indent=2)

        logger.info("Saved evaluation results to caption_eval_results.json")
        return results


    @backoff.on_exception(
        backoff.constant,
        # here you should add more model exceptions as you want,
        # but you are forbidden to add "Exception", that is, a common type of exception
        # because we want to catch this kind of Exception in the outside to ensure each example won't exceed the time limit
        (
                # General exceptions
                SSLError,

                # OpenAI exceptions
                openai.RateLimitError,
                openai.BadRequestError,
                openai.InternalServerError,

                # Google exceptions
                InvalidArgument,
                ResourceExhausted,
                InternalServerError,
                BadRequest,

                # Groq exceptions
                # todo: check
        ),
        interval=30,
        max_tries=10
    )
    def load_vqa_answers(
        self,
        reference_answer_file: str,
        reference_question_file: str,
        generated_answer_file: str
    ) -> List[tuple[int, List[str], str, List[Dict], str, str]]:  # 返回增加answer_type

        with open(reference_question_file, 'r') as f:
            question_data = json.load(f)["questions"]
        with open(reference_answer_file, 'r') as f:
            answer_data = json.load(f)["annotations"]
        with open(generated_answer_file, 'r') as f:
            generated_data = json.load(f)

        reference_map = {}
        for qa in answer_data:
            qid = qa["question_id"]
            answers = [a["answer"] for a in qa["answers"]]
            reference_map[qid] = {
                "answers": answers,
                "question_type": qa.get("question_type", ""),
                "answer_type": qa.get("answer_type", "")  # 新增answer_type
            }

        for q in question_data:
            qid = q["question_id"]
            if qid in reference_map:
                reference_map[qid]["question"] = q["question"]

        results = []
        for item in generated_data:
            qid = item["question_id"]
            generated_answer = item["answer"]

            if qid not in reference_map or "question" not in reference_map[qid]:
                logger.warning(f"Missing reference for question_id: {qid}")
                continue

            ref = reference_map[qid]
            question = ref["question"]
            human_answers = ref["answers"]
            question_type = ref.get("question_type", "")
            answer_type = ref.get("answer_type", "")  # 新增answer_type获取

            prompt = (
                "Question: {}\n"
                "Generated Answer: {}\n"
                "Human Answers: {}\n"
            ).format(question, generated_answer, "; ".join(human_answers))

            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
            # 返回值元组加上answer_type
            results.append((qid, human_answers, generated_answer, messages, question_type, answer_type))
        return results


    def predict_vqa_accuracy(
        self,
        reference_question_file: str,
        reference_answer_file: str,
        generated_answer_file: str,
        output_file: str = "vqa_eval_results.json"
    ) -> Dict:

        eval_data = self.load_vqa_answers(reference_question_file, reference_answer_file, generated_answer_file)

        all_results = []
        for qid, human_answers, gen_answer, messages, question_type, answer_type in eval_data:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens
            }

            try:
                response = self.call_llm(payload)
                logger.info(f"Response for {qid}: {response}")
                score = float(response.strip())
            except Exception as e:
                logger.error(f"LLM call failed for {qid}: {e}")
                score = None
            except ValueError:
                logger.warning(f"Invalid score format for {qid}: {response}")
                score = None

            all_results.append({
                "question_id": qid,
                "question_type": question_type,
                "answer_type": answer_type,  # 新增answer_type字段
                "generated_answer": gen_answer,
                "human_answers": human_answers,
                "vqa_accuracy": score
            })

        valid_scores = [r["vqa_accuracy"] for r in all_results if r["vqa_accuracy"] is not None]
        average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        output = {
            "average_accuracy": average_score,
            "individual_results": all_results
        }

        return output


    def load_reference_with_images(self,reference_file, img_path):
        import json
        import os

        with open(reference_file, 'r') as f:
            reference_data = json.load(f)

        all_messages = []
        for obj in reference_data:
            # 获取图像相对路径和 image_id
            relative_path = obj["image"]
            image_id = int(relative_path.split("_")[-1].split(".")[0])
            full_image_path = os.path.join(img_path, relative_path)

            # 合并多参考caption为单个字符串
            reference_caption = " ".join(obj["caption"])
            messages = [
                {
                    "role": "system",
                    "content": self.system_message
                },
                {
                    "role": "user",
                    "content": (
                        f"**Captions:** {reference_caption}"
                    ),
                    "image": self.encode_image_to_base64(full_image_path)  # 提供本地图像路径
                }
            ]
            all_messages.append((image_id, reference_caption, messages))

        return all_messages

    def predict_multimodal(self, reference_file: str, image_path: str) -> List[Dict]:
        """
        Perform batch evaluation of image-caption semantic similarity using local images.
        """
        message_data = self.load_reference_with_images(reference_file, image_path)
        # self.check_and_print_gpt4o_messages_format(message_data)
        # exit(1)
        results = []
        for image_id, ref_caption, messages in message_data:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            }

            try:
                response = self.call_llm(payload)
                logger.info("RESPONSE: %s", response)
                score = float(response.strip())
            except Exception as e:
                logger.error("Failed to call LLM: %s", str(e))
                score = None
            except ValueError:
                logger.warning("Invalid score returned: %s", response)
                score = None

            results.append({
                "image_id": image_id,
                "ref_caption": ref_caption,
                "score": score
            })

        valid_scores = [r["score"] for r in results if r["score"] is not None]
        average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        output = {
            "average_score": average_score,
            "results": results
        }

        with open("caption_eval_results.json", "w") as f:
            json.dump(output, f, indent=2)

        logger.info("Saved evaluation results to caption_eval_results.json")
        return results    
    def call_llm(self, payload):
        if self.model.startswith(("gpt", "o")):
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {require_env('OPENAI_API_KEY')}"
            }
            logger.info("Generating content with GPT model: %s", self.model)
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                error_payload = response.json()
                if error_payload.get('error', {}).get('code') == "context_length_exceeded":
                    logger.error("Context length exceeded. Retrying with a smaller context.")
                    payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]
                    retry_response = requests.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    if retry_response.status_code != 200:
                        logger.error(
                            "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)
                        return ""

                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                return response.json()['choices'][0]['message']['content']

        elif self.model.startswith("claude"):
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            claude_messages = []

            for i, message in enumerate(messages):
                claude_message = {
                    "role": message["role"],
                    "content": []
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"
                for part in message["content"]:

                    if part['type'] == "image_url":
                        image_source = {}
                        image_source["type"] = "base64"
                        image_source["media_type"] = "image/png"
                        image_source["data"] = part['image_url']['url'].replace("data:image/png;base64,", "")
                        claude_message['content'].append({"type": "image", "source": image_source})

                    if part['type'] == "text":
                        claude_message['content'].append({"type": "text", "text": part['text']})

                claude_messages.append(claude_message)

            # the claude not support system message in our endpoint, so we concatenate it at the first user message
            if claude_messages[0]['role'] == "system":
                claude_system_message_item = claude_messages[0]['content'][0]
                claude_messages[1]['content'].insert(0, claude_system_message_item)
                claude_messages.pop(0)

            logger.debug("CLAUDE MESSAGE: %s", repr(claude_messages))

            headers = {
                "x-api-key": require_env("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": claude_messages,
                "temperature": temperature,
                "top_p": top_p
            }

            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:

                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                return response.json()['content'][0]['text']

        elif self.model.startswith("mistral"):
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            assert self.observation_type in pure_text_settings, f"The model {self.model} can only support text-based input, please consider change based model or settings"

            mistral_messages = []

            for i, message in enumerate(messages):
                mistral_message = {
                    "role": message["role"],
                    "content": ""
                }

                for part in message["content"]:
                    mistral_message['content'] = part['text'] if part['type'] == "text" else ""

                mistral_messages.append(mistral_message)

            from openai import OpenAI

            client = OpenAI(api_key=require_env("TOGETHER_API_KEY"),
                            base_url='https://api.together.xyz',
                            )

            flag = 0
            while True:
                try:
                    if flag > 20:
                        break
                    logger.info("Generating content with model: %s", self.model)
                    response = client.chat.completions.create(
                        messages=mistral_messages,
                        model=self.model,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        temperature=temperature
                    )
                    break
                except:
                    if flag == 0:
                        mistral_messages = [mistral_messages[0]] + mistral_messages[-1:]
                    else:
                        mistral_messages[-1]["content"] = ' '.join(mistral_messages[-1]["content"].split()[:-500])
                    flag = flag + 1

            try:
                return response.choices[0].message.content
            except Exception as e:
                print("Failed to call LLM: " + str(e))
                return ""

        elif self.model.startswith("THUDM"):
            # THUDM/cogagent-chat-hf
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            cog_messages = []

            for i, message in enumerate(messages):
                cog_message = {
                    "role": message["role"],
                    "content": []
                }

                for part in message["content"]:
                    if part['type'] == "image_url":
                        cog_message['content'].append(
                            {"type": "image_url", "image_url": {"url": part['image_url']['url']}})

                    if part['type'] == "text":
                        cog_message['content'].append({"type": "text", "text": part['text']})

                cog_messages.append(cog_message)

            # the cogagent not support system message in our endpoint, so we concatenate it at the first user message
            if cog_messages[0]['role'] == "system":
                cog_system_message_item = cog_messages[0]['content'][0]
                cog_messages[1]['content'].insert(0, cog_system_message_item)
                cog_messages.pop(0)

            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": cog_messages,
                "temperature": temperature,
                "top_p": top_p
            }

            base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000").rstrip("/")

            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, stream=False)
            if response.status_code == 200:
                decoded_line = response.json()
                content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
                return content
            else:
                print("Failed to call LLM: ", response.status_code)
                return ""

        elif self.model in ["gemini-pro", "gemini-pro-vision"]:
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            if self.model == "gemini-pro":
                assert self.observation_type in pure_text_settings, f"The model {self.model} can only support text-based input, please consider change based model or settings"

            gemini_messages = []
            for i, message in enumerate(messages):
                role_mapping = {
                    "assistant": "model",
                    "user": "user",
                    "system": "system"
                }
                gemini_message = {
                    "role": role_mapping[message["role"]],
                    "parts": []
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"

                # The gemini only support the last image as single image input
                if i == len(messages) - 1:
                    for part in message["content"]:
                        gemini_message['parts'].append(part['text']) if part['type'] == "text" \
                            else gemini_message['parts'].append(encoded_img_to_pil_img(part['image_url']['url']))
                else:
                    for part in message["content"]:
                        gemini_message['parts'].append(part['text']) if part['type'] == "text" else None

                gemini_messages.append(gemini_message)

            # the gemini not support system message in our endpoint, so we concatenate it at the first user message
            if gemini_messages[0]['role'] == "system":
                gemini_messages[1]['parts'][0] = gemini_messages[0]['parts'][0] + "\n" + gemini_messages[1]['parts'][0]
                gemini_messages.pop(0)

            # since the gemini-pro-vision donnot support multi-turn message
            if self.model == "gemini-pro-vision":
                message_history_str = ""
                for message in gemini_messages:
                    message_history_str += "<|" + message['role'] + "|>\n" + message['parts'][0] + "\n"
                gemini_messages = [{"role": "user", "parts": [message_history_str, gemini_messages[-1]['parts'][1]]}]
                # gemini_messages[-1]['parts'][1].save("output.png", "PNG")

            # print(gemini_messages)
            genai.configure(api_key=require_env("GENAI_API_KEY"))
            logger.info("Generating content with Gemini model: %s", self.model)
            request_options = {"timeout": 120}
            gemini_model = genai.GenerativeModel(self.model)

            response = gemini_model.generate_content(
                gemini_messages,
                generation_config={
                    "candidate_count": 1,
                    # "max_output_tokens": max_tokens,
                    "top_p": top_p,
                    "temperature": temperature
                },
                safety_settings={
                    "harassment": "block_none",
                    "hate": "block_none",
                    "sex": "block_none",
                    "danger": "block_none"
                },
                request_options=request_options
            )
            return response.text

        elif self.model == "gemini-1.5-pro-latest":
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            gemini_messages = []
            for i, message in enumerate(messages):
                role_mapping = {
                    "assistant": "model",
                    "user": "user",
                    "system": "system"
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"
                gemini_message = {
                    "role": role_mapping[message["role"]],
                    "parts": []
                }

                # The gemini only support the last image as single image input
                for part in message["content"]:

                    if part['type'] == "image_url":
                        # Put the image at the beginning of the message
                        gemini_message['parts'].insert(0, encoded_img_to_pil_img(part['image_url']['url']))
                    elif part['type'] == "text":
                        gemini_message['parts'].append(part['text'])
                    else:
                        raise ValueError("Invalid content type: " + part['type'])

                gemini_messages.append(gemini_message)

            # the system message of gemini-1.5-pro-latest need to be inputted through model initialization parameter
            system_instruction = None
            if gemini_messages[0]['role'] == "system":
                system_instruction = gemini_messages[0]['parts'][0]
                gemini_messages.pop(0)

            genai.configure(api_key=require_env("GENAI_API_KEY"))
            logger.info("Generating content with Gemini model: %s", self.model)
            request_options = {"timeout": 120}
            gemini_model = genai.GenerativeModel(
                self.model,
                system_instruction=system_instruction
            )

            with open("response.json", "w") as f:
                messages_to_save = []
                for message in gemini_messages:
                    messages_to_save.append({
                        "role": message["role"],
                        "content": [part if isinstance(part, str) else "image" for part in message["parts"]]
                    })
                json.dump(messages_to_save, f, indent=4)

            response = gemini_model.generate_content(
                gemini_messages,
                generation_config={
                    "candidate_count": 1,
                    # "max_output_tokens": max_tokens,
                    "top_p": top_p,
                    "temperature": temperature
                },
                safety_settings={
                    "harassment": "block_none",
                    "hate": "block_none",
                    "sex": "block_none",
                    "danger": "block_none"
                },
                request_options=request_options
            )

            return response.text

        elif self.model == "llama3-70b":
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            assert self.observation_type in pure_text_settings, f"The model {self.model} can only support text-based input, please consider change based model or settings"

            groq_messages = []

            for i, message in enumerate(messages):
                groq_message = {
                    "role": message["role"],
                    "content": ""
                }

                for part in message["content"]:
                    groq_message['content'] = part['text'] if part['type'] == "text" else ""

                groq_messages.append(groq_message)

            # The implementation based on Groq API
            client = Groq(
                api_key=require_env("GROQ_API_KEY"),
            )

            flag = 0
            while True:
                try:
                    if flag > 20:
                        break
                    logger.info("Generating content with model: %s", self.model)
                    response = client.chat.completions.create(
                        messages=groq_messages,
                        model="llama3-70b-8192",
                        max_tokens=max_tokens,
                        top_p=top_p,
                        temperature=temperature
                    )
                    break
                except:
                    if flag == 0:
                        groq_messages = [groq_messages[0]] + groq_messages[-1:]
                    else:
                        groq_messages[-1]["content"] = ' '.join(groq_messages[-1]["content"].split()[:-500])
                    flag = flag + 1

            try:
                return response.choices[0].message.content
            except Exception as e:
                print("Failed to call LLM: " + str(e))
                return ""

        elif self.model.startswith("qwen"):
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            qwen_messages = []

            for i, message in enumerate(messages):
                qwen_message = {
                    "role": message["role"],
                    "content": []
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"
                for part in message["content"]:
                    qwen_message['content'].append(
                        {"image": "file://" + save_to_tmp_img_file(part['image_url']['url'])}) if part[
                                                                                                      'type'] == "image_url" else None
                    qwen_message['content'].append({"text": part['text']}) if part['type'] == "text" else None

                qwen_messages.append(qwen_message)

            flag = 0
            while True:
                try:
                    if flag > 20:
                        break
                    logger.info("Generating content with model: %s", self.model)

                    if self.model in ["qwen-vl-plus", "qwen-vl-max"]:
                        response = dashscope.MultiModalConversation.call(
                            model=self.model,
                            messages=qwen_messages,
                            result_format="message",
                            max_length=max_tokens,
                            top_p=top_p,
                            temperature=temperature
                        )

                    elif self.model in ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-max-0428", "qwen-max-0403",
                                        "qwen-max-0107", "qwen-max-longcontext"]:
                        response = dashscope.Generation.call(
                            model=self.model,
                            messages=qwen_messages,
                            result_format="message",
                            max_length=max_tokens,
                            top_p=top_p,
                            temperature=temperature
                        )

                    else:
                        raise ValueError("Invalid model: " + self.model)

                    if response.status_code == HTTPStatus.OK:
                        break
                    else:
                        logger.error('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                            response.request_id, response.status_code,
                            response.code, response.message
                        ))
                        raise Exception("Failed to call LLM: " + response.message)
                except:
                    if flag == 0:
                        qwen_messages = [qwen_messages[0]] + qwen_messages[-1:]
                    else:
                        for i in range(len(qwen_messages[-1]["content"])):
                            if "text" in qwen_messages[-1]["content"][i]:
                                qwen_messages[-1]["content"][i]["text"] = ' '.join(
                                    qwen_messages[-1]["content"][i]["text"].split()[:-500])
                    flag = flag + 1

            try:
                if self.model in ["qwen-vl-plus", "qwen-vl-max"]:
                    return response['output']['choices'][0]['message']['content'][0]['text']
                else:
                    return response['output']['choices'][0]['message']['content']

            except Exception as e:
                print("Failed to call LLM: " + str(e))
                return ""

        else:
            raise ValueError("Invalid model: " + self.model)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def require_arg(parser: argparse.ArgumentParser, value: str, name: str) -> None:
    if not value:
        parser.error(f"{name} is required for this mode")

def inspect_loaded_messages(reference_file: str, generated_file: str, agent: PromptAgent, sample_size: int = 1):
    """
    检查 PromptAgent.load_captions 是否正确生成 message 格式数据。
    显示前 sample_size 条消息样本。
    """
    messages = agent.load_captions(reference_file, generated_file)
    if not messages:
        print("⚠️ 没有加载到任何消息！请检查输入文件路径或格式。")
        return

    print(f"\n✅ 成功加载 {len(messages)} 条 message 数据，展示前 {sample_size} 条：\n")
    for i, (image_id, ref_caption, gen_caption, msg) in enumerate(messages[:sample_size]):
        print(f"[Image ID] {image_id}")
        print(f"[Reference Caption] {ref_caption}")
        print(f"[Generated Caption] {gen_caption}")
        print(f"[System Message]\n{msg[0]['content'][:200]}...\n")
        print(f"[User Prompt]\n{msg[1]['content']}\n{'-'*60}")


if __name__ == "__main__":


    parser = build_parser()
    args = parser.parse_args()

    agent = PromptAgent(
        model=args.model
    )

    if args.inspect:
        require_arg(parser, args.gen, "--gen")
        inspect_loaded_messages(args.ref, args.gen, agent)
    elif args.multimodal:
        require_arg(parser, args.img_path, "--img_path")
        # 自动生成时间戳路径
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"./gpt4o_judge/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "result.json")

        results = agent.predict_multimodal(args.ref, args.img_path)

        with open(output_path, "w") as f:
            json.dump({
                "average_score": sum(r["score"] for r in results if r["score"] is not None) / max(len(results), 1),
                "results": results
            }, f, indent=2)

        print(f"\n✅ 已保存结果至：{output_path}")    
    elif args.vqa:
        require_arg(parser, args.gen_answer, "--gen_answer")
        # 自动生成时间戳路径
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"./gpt4o_judge/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "result.json")

        results = agent.predict_vqa_accuracy(args.ref_answer, args.ref_question, args.gen_answer)

        output = {
            "average_accuracy": results[ "average_accuracy"],
            "individual_results": results["individual_results"]
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved VQA evaluation results to {output_path}")
        print(f"\n✅ 已保存结果至：{output_path}")                
    else:
        require_arg(parser, args.gen, "--gen")
        # 自动生成时间戳路径
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"./gpt4o_judge/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "result.json")

        results = agent.predict(args.ref, args.gen)

        with open(output_path, "w") as f:
            json.dump({
                "average_score": sum(r["score"] for r in results if r["score"] is not None) / max(len(results), 1),
                "results": results
            }, f, indent=2)

        print(f"\n✅ 已保存结果至：{output_path}")
