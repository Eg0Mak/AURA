import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc


class QueryExpander:
    """
    Модуль для генерации дополнительных формулировок запроса:
    - Режим 'expand'  — расширение запроса (добавление контекста);
    - Режим 'paraphrase' — перефразирование (изменение формулировки);
    """

    def __init__(
            self,
            model_name: str = "l3lab/L1-Qwen3-8B-Max",
            device: str = "cuda",
            max_new_tokens: int = 75,
            load_in_8bit: bool = True,
            load_in_4bit: bool = False
    ):
        print(f"Загружается модель {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )

        self.device = device
        self.max_new_tokens = max_new_tokens

    # ---------------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """
        Генерация без утечки <think>.
        """

        messages = [
            # IMPORTANT: chain-of-thought is forbidden
            {
                "role": "system",
                "content": (
                    "Respond only with the final result. "
                    "Do not reveal your reasoning process, chain-of-thought, reasoning tokens, "
                    "and do not use <think> tags. "
                    "Answer briefly and without additional comments."
                )
            },
            {"role": "user", "content": prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.05
            )

        text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip().replace("<think>", "").replace("</think>", "")

        # чистим GPU
        torch.cuda.empty_cache()
        gc.collect()

        return text

        text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        return text.strip()

    def _build_expand_prompt(self, query: str, n_variants: int) -> str:
        return f"""
    You must output ONLY the final expanded variants.  
    The output MUST contain nothing except the final expanded sentences.  

    ABSOLUTELY FORBIDDEN under any circumstances:  
    - any reasoning  
    - any explanations  
    - any analysis  
    - any chain-of-thought  
    - any assumptions  
    - any comments  
    - any meta-text  
    - any mention of what you are doing  
    - any reference to rules or instructions  
    - any introductions, transitions, or filler phrases  
    - any statements about the original query  
    - any statements about generating variants  
    - any summaries, clarifications, or descriptions  
    - ANYTHING except the final expanded variants in Russian  

    If your answer contains even one extra word beyond the final sentences, the output is INVALID.

    You are an intelligent assistant expanding the original query by adding relevant details, clarifications, related aspects, and real usage scenarios — WITHOUT changing the query’s purpose and WITHOUT producing any reasoning text.

    ### STRICT RULES:
    1. ❗️You MUST NOT change the purpose of the original query.  
    2. ❗️You MUST NOT paraphrase — you must ADD context.  
    3. ❗️Every variant must introduce NEW, concrete information.  
    4. ❗️Each variant must add something meaningful: action, condition, limitation, example, or user motivation.  
    5. ❗️Do NOT remove important information.  
    6. ❗️Do NOT output thoughts, explanations, or any text except final sentences.  
    7. ❗️Your entire output must consist ONLY of the final Russian expanded variants, one per line.

    Each variant must be natural, detailed, and written in Russian.

    ---

    ### Примеры

    Пример 1  
    Исходный запрос: "Как оплатить кредит?"  
    Расширенные версии:  
    - "Как оплатить кредит через мобильное приложение банка, если нет доступа к интернет-банку?"  
    - "Какие способы доступны для оплаты кредита в выходные или праздничные дни?"  
    - "Можно ли оплатить кредит досрочно с другой карты, не своего банка?"  

    Пример 2  
    Исходный запрос: "Проблемы с входом в личный кабинет"  
    Расширенные версии:  
    - "Почему не получается войти в личный кабинет после смены пароля?"  
    - "Что делать, если при входе появляется ошибка 'неверный код подтверждения'?"  
    - "Как восстановить доступ, если утерян номер телефона, привязанный к входу?"  

    Пример 3  
    Исходный запрос: "Как получить карту?"  
    Расширенные версии:  
    - "Как заказать банковскую карту онлайн и получить её на дом?"  
    - "Какие документы нужны, чтобы получить карту в офисе банка?"  
    - "Можно ли оформить карту для несовершеннолетнего ребёнка?"  

    ---

    Now process the following query:

    Original query: {query}

    Generate {n_variants} expanded variants.  
    Each variant must appear on a new line without numbering.  
    Do NOT output anything except the final expanded sentences.
    """

    def _build_rephrase_prompt(self, query: str, n_variants: int) -> str:
        return f"""
        You must output ONLY the final rephrased variants.  
        Output MUST consist exclusively of the {n_variants} final lines.  
        Forbidden: any introductions, explanations, reasoning, internal thoughts, comments, analysis, meta-text, or references to the task.  
        If you attempt to explain anything, the output is considered invalid.  
        Your response must contain NOTHING except the rephrased variants.

        You are an assistant rephrasing the user's query while preserving its meaning and changing its wording, tone, and structure.  
        Do not add new information. Do not omit important information.  
        You MUST preserve all key meaning components and MUST NOT change or distort the original intent.  
        All key terms and core semantic elements MUST remain present in each variant.

        ---

        ### Примеры (few-shot)

        Пример 1  
        Исходный запрос: "Как оплатить кредит?"  
        Перефразированные версии:  
        - "Какие способы оплаты кредита доступны?"  
        - "Как можно внести платеж по кредиту?"  
        - "Каким образом оплатить кредитный долг?"  

        Пример 2  
        Исходный запрос: "Проблемы с входом в личный кабинет"  
        Перефразированные версии:  
        - "Не получается войти в личный кабинет"  
        - "Ошибка при попытке входа в личный кабинет"  
        - "Почему не удается авторизоваться в личном кабинете?"  

        Пример 3  
        Исходный запрос: "Как получить карту?"  
        Перефразированные версии:  
        - "Что нужно, чтобы оформить карту?"  
        - "Как оформить и получить банковскую карту?"  
        - "Каким образом можно заказать карту?"  

        ---

        Original query: {query}

        Generate {n_variants} variants.  
        Each variant must be on its own line.  
        Output NOTHING except these final lines.
        """

    def expand_query(self, query: str, n_variants: int = 3) -> list[str]:
        prompt = self._build_expand_prompt(query, n_variants)
        text = self._generate(prompt)
        variants = [t.strip() for t in text.split("\n") if t.strip()]
        return variants[:n_variants]

    def paraphrase_query(self, query: str, n_variants: int = 3) -> list[str]:
        prompt = self._build_rephrase_prompt(query, n_variants)
        text = self._generate(prompt)
        variants = [t.strip() for t in text.split("\n") if t.strip()]
        return variants[:n_variants]


    def generate(self, query: str, mode: str = "expand", n_variants: int = 3) -> list[str]:
        if mode == "expand":
            return self.expand_query(query, n_variants)
        elif mode == "paraphrase":
            return self.paraphrase_query(query, n_variants)
        else:
            raise ValueError("mode должен быть 'expand' или 'paraphrase'")