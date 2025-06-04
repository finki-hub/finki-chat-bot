from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.data.db import get_db
from app.data.questions import get_closest_questions
from app.llms.embeddings import generate_embeddings
from app.llms.prompts import build_context, build_prompt
from app.llms.responses import generate_response
from app.schemas.chat import ChatRequestSchema

db_dep = Depends(get_db)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    dependencies=[db_dep],
)


@router.post(
    "/",
    summary="Stream a chat response",
    description=(
        "Compute an embedding for the incoming question, retrieve top-N "
        "similar questions for context, construct a prompt, and stream back "
        "the LLM's answer as a text stream."
    ),
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Unsupported model or invalid request"},
    },
    operation_id="chatWithModel",
)
async def chat(
    opts: ChatRequestSchema,
    db: Database = db_dep,
) -> StreamingResponse:
    question_emb = await generate_embeddings(opts.question, opts.embeddings_model)

    relevant_questions = await get_closest_questions(
        db,
        question_emb,
        opts.embeddings_model,
        limit=20,
    )

    context = build_context(relevant_questions)
    prompt = build_prompt(context, opts.question)

    return await generate_response(prompt, opts.inference_model)
