# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODELNAME = 'Qwen/Qwen2.5-0.5B-Instruct' #"Qwen/Qwen3-4B-Instruct-2507"

class LLMAgent:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODELNAME,
            use_fast=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODELNAME,
            dtype=torch.float32,
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
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])