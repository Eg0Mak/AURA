def build_prompt(context: str, question: str):
    """
    Формирует системный и пользовательский промпты для RAG-системы в режиме QA (Question Answering).
    Цель — получить точный, достоверный и лаконичный ответ исключительно на основе контекста.
    """
    system_message = {
        "role": "system",
        "content": (
            "You are an expert assistant working in a Retrieval-Augmented Generation (RAG) QA system. "
            "Your goal is to answer the user's question **strictly and only** using the information from the provided CONTEXT.\n\n"
            "Follow these precise rules:\n"
            "1. Respond **in Russian**.\n"
            "2. Base your answer **only on the given context** — do not use prior knowledge, common sense, or external facts.\n"
            "3. If the context does not contain enough information to answer the question, reply exactly with:\n"
            "   'К сожалению, в предоставленном контексте недостаточно информации для ответа на ваш вопрос.'\n"
            "4. Keep your response **short, clear, and factual** — 1–3 sentences.\n"
            "5. If several parts of the context provide different information, mention this briefly (e.g., 'В контексте приведены противоречивые данные...').\n"
            "6. Do not summarize the entire context — answer **specifically** to the user's question.\n"
            "7. Do not make assumptions, predictions, or subjective judgments.\n"
            "8. Maintain a neutral and professional tone in Russian."
        )
    }

    user_message = {
        "role": "user",
        "content": (
            f"=== КОНТЕКСТ ===\n{context.strip()}\n\n"
            f"=== ВОПРОС ===\n{question.strip()}\n\n"
            "=== ОТВЕТ ==="
        )
    }

    return [system_message, user_message]
