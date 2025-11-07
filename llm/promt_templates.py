def build_prompt(context: str, question: str):
    """
    Формирует системный и пользовательский промпт для LLM.
    Контекст - это retrieved текст из RAG (векторной базы).
    """
    system_message = {
        "role": "system",
        "content": (
            "Ты - умный и вежливый помощник, который отвечает на русском языке. "
            "Используй только приведённую информацию из контекста. "
            "Если ответа нет в контексте — честно скажи, что данных недостаточно."
        ),
    }

    user_message = {
        "role": "user",
        "content": (
            f"Контекст:\n{context}\n\n"
            f"Вопрос: {question}\n\n"
            "Ответ:"
        ),
    }

    return [system_message, user_message]
