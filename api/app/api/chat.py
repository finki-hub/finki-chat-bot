from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.data.questions import get_closest_questions
from app.llms.embeddings import generate_embeddings
from app.llms.ollama import stream_ollama_response
from app.llms.prompts import build_context, build_prompt
from app.schema.chat import ChatQuestion

router = APIRouter(tags=["Chat"])


@router.post("/")
async def chat(options: ChatQuestion) -> StreamingResponse:
    user_embedding = await generate_embeddings(options.question, options.model)
    questions = await get_closest_questions(user_embedding, options.model)

    context = build_context(questions)
    prompt = build_prompt(context, options.question)

    return await stream_ollama_response(prompt, options.model)
