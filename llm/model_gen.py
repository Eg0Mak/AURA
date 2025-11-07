from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import torch
import os
from llm.prompt_templates import build_prompt

load_dotenv()

MODEL_NAME = os.getenv("LLM_MODEL", "HuggingFaceTB/SmolLM3-3B")


class LLMGenerator:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_answer(self, context: str, question: str, max_tokens: int = 200) -> str:
        """
        Генерация ответа с учётом retrieved контекста.
        """
        prompt_messages = build_prompt(context, question)

        inputs = self.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        return answer.strip()
