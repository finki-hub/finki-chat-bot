from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.data.questions import get_closest_questions
from app.llms.embeddings import generate_embeddings
from app.llms.prompts import build_context, build_prompt
from app.llms.responses import generate_response
from app.schema.chat import ChatQuestion

router = APIRouter(tags=["Chat"])


@router.post("/")
async def chat(options: ChatQuestion) -> StreamingResponse:
    question_embedding = await generate_embeddings(
        options.question,
        options.embeddings_model,
    )
    questions = await get_closest_questions(
        question_embedding,
        options.embeddings_model,
        limit=12,
    )

    context = build_context(questions)
    prompt = build_prompt(context, options.question)

    return await generate_response(prompt, options.embeddings_model)
