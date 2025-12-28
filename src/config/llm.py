# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMAgent:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507",
            use_fast=True
        ) # "Qwen/Qwen3-0.6B")  # "google/functiongemma-270m-it"

        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507",
            dtype=torch.float16,
            low_cpu_mem_usage=True

        )

        self.model.eval()

    def answer(self, message):
        inputs = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to('cpu')

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])