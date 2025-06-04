import urllib.parse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.constants import strings
from app.constants.defaults import DEFAULT_EMBEDDINGS_MODEL
from app.constants.errors import QUESTION_404
from app.data.questions import (
    create_question_query,
    delete_question_query,
    get_closest_questions,
    get_nth_question_query,
    get_question_by_name_query,
    get_question_names_query,
    get_questions_query,
    update_question_query,
)
from app.llms.embeddings import fill_embeddings, generate_embeddings
from app.llms.models import Model
from app.schema.embedding import EmbeddingOptions
from app.schema.question import (
    CreateQuestionSchema,
    QuestionSchema,
    UpdateQuestionSchema,
)
from app.utils.auth import verify_api_key

router = APIRouter(tags=["Questions"])


@router.get("/list", response_model=list[QuestionSchema])
async def get_questions() -> list[QuestionSchema]:
    result = await get_questions_query()

    return result


@router.get("/names", response_model=list[str])
async def get_question_names() -> list[str]:
    result = await get_question_names_query()

    return result


@router.get("/name/{name:path}", response_model=QuestionSchema)
async def get_question_by_name(name: str) -> QuestionSchema:
    decoded_name = urllib.parse.unquote(name)

    result = await get_question_by_name_query(decoded_name)

    if not result:
        raise HTTPException(status_code=404, detail=QUESTION_404)

    return result


@router.post("/create", response_model=QuestionSchema)
async def create_question(question: CreateQuestionSchema) -> QuestionSchema:
    existing_question = await get_question_by_name_query(question.name)

    if existing_question:
        raise HTTPException(status_code=400, detail="Question already exists")

    result = await create_question_query(question)

    if not result:
        raise HTTPException(status_code=400, detail="Failed to create question")

    return result


@router.put("/update/{name:path}", response_model=QuestionSchema)
async def update_question(name: str, question: UpdateQuestionSchema) -> QuestionSchema:
    decoded_name = urllib.parse.unquote(name)
    existing_question = await get_question_by_name_query(decoded_name)

    if not existing_question:
        raise HTTPException(status_code=404, detail=QUESTION_404)

    updates = question.model_dump(exclude_unset=True)

    if len(updates) == 0:
        raise HTTPException(status_code=400, detail="No updates provided")

    result = await update_question_query(decoded_name, question)

    if not result:
        raise HTTPException(status_code=400, detail="Failed to update question")

    return result


@router.delete("/delete/{name:path}", response_model=QuestionSchema)
async def delete_question(name: str) -> QuestionSchema:
    decoded_name = urllib.parse.unquote(name)
    existing_question = await get_question_by_name_query(decoded_name)

    if not existing_question:
        raise HTTPException(status_code=404, detail=QUESTION_404)

    await delete_question_query(decoded_name)

    return existing_question


@router.get("/nth/{n}", response_model=QuestionSchema)
async def get_nth_question(n: int) -> QuestionSchema:
    result = await get_nth_question_query(n)

    if not result:
        raise HTTPException(status_code=404, detail=QUESTION_404)

    return result


@router.post("/embed", response_model=str)
async def embed_questions(
    options: EmbeddingOptions,
    _: None = Depends(verify_api_key),
) -> str:
    await fill_embeddings(options.model, options.all)

    return strings.EMBEDDING_SUCCESS


class ClosestRequestSchema(BaseModel):
    question: str
    embeddings_model: Model = Field(default=DEFAULT_EMBEDDINGS_MODEL)


@router.get("/closest")
async def get_closest_questions_endpoint(
    options: ClosestRequestSchema,
) -> list[str]:
    question_embedding = await generate_embeddings(
        options.question,
        options.embeddings_model,
    )
    questions = await get_closest_questions(
        question_embedding,
        options.embeddings_model,
        limit=20,
    )

    return [q.name for q in questions]
