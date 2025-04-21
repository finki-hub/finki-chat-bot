from fastapi import APIRouter

from app.data.questions import get_closest_questions
from app.llms.ollama import generate_ollama_response
from app.llms.utils import generate_embeddings
from app.schema.chat import ChatQuestion

router = APIRouter(tags=["Chat"])


@router.post("/")
async def chat(options: ChatQuestion) -> dict[str, str]:
    user_embedding = generate_embeddings(options.question, options.model)

    questions = await get_closest_questions(user_embedding, options.model)

    if not questions:
        return {"message": "Не знам."}  # noqa: RUF001

    context = "\n".join(
        [f"Наслов: {q.name}\nСодржина: {q.content}" for q in questions],  # noqa: RUF001
    )

    prompt = f"Контекст:\n{context}\n\nПрашање: {options.question}\nОдговор:"  # noqa: RUF001

    return {"answer": generate_ollama_response(prompt, options.model)}
