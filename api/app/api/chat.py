from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.data.questions import get_closest_questions
from app.llms.embeddings import generate_embeddings
from app.llms.models import Model
from app.llms.prompts import build_context, build_prompt
from app.llms.responses import generate_response

router = APIRouter(tags=["Chat"])


class ChatRequestSchema(BaseModel):
    question: str
    embeddings_model: Model = Field(default=Model.BGE_M3)
    inference_model: Model = Field(default=Model.MISTRAL)


@router.post("/")
async def chat(options: ChatRequestSchema) -> StreamingResponse:
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
