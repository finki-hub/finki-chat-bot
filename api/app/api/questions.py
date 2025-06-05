import urllib.parse

from fastapi import APIRouter, Depends, HTTPException, status

from app.data.connection import Database
from app.data.db import get_db
from app.data.questions import (
    create_question_query,
    delete_question_query,
    get_nth_question_query,
    get_question_by_name_query,
    get_question_names_query,
    get_questions_query,
    update_question_query,
)
from app.data.questions import (
    get_closest_questions as query_closest_questions,
)
from app.llms.embeddings import fill_embeddings, generate_embeddings
from app.schemas.questions import (
    CreateQuestionSchema,
    EmbedQuestionsSchema,
    GetClosestQuestionsSchema,
    QuestionSchema,
    UpdateQuestionSchema,
)
from app.utils.auth import verify_api_key

db_dep = Depends(get_db)
api_key_dep = Depends(verify_api_key)

router = APIRouter(
    prefix="/questions",
    tags=["Questions"],
    dependencies=[db_dep],
)


@router.get(
    "/",
    summary="List all questions",
    description="Returns a list of all stored questions.",
    response_model=list[QuestionSchema],
    status_code=status.HTTP_200_OK,
    operation_id="listQuestions",
)
async def list_questions(db: Database = db_dep) -> list[QuestionSchema]:
    return await get_questions_query(db)


@router.get(
    "/names",
    summary="List question names",
    description="Returns only the names of all stored questions.",
    response_model=list[str],
    status_code=status.HTTP_200_OK,
    operation_id="listQuestionNames",
)
async def list_question_names(db: Database = db_dep) -> list[str]:
    return await get_question_names_query(db)


@router.get(
    "/closest",
    summary="Find closest questions",
    description="Given a query and an embedding model, return the top N closest question names.",
    response_model=list[QuestionSchema],
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "No questions found"}},
    operation_id="getClosestQuestions",
)
async def closest_questions(
    params: GetClosestQuestionsSchema = Depends(),  # noqa: B008
    db: Database = db_dep,
) -> list[QuestionSchema]:
    prompt_embedding = await generate_embeddings(params.prompt, params.model)
    results = await query_closest_questions(
        db,
        prompt_embedding,
        params.model,
        limit=params.limit,
        threshold=params.threshold,
    )
    return results


@router.get(
    "/name/{name:path}",
    summary="Get question by name",
    description="Return the matching question, 404 if not found.",
    response_model=QuestionSchema,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Question not found"}},
    operation_id="getQuestionByName",
)
async def get_question_by_name(
    name: str,
    db: Database = db_dep,
) -> QuestionSchema:
    decoded = urllib.parse.unquote(name)
    question = await get_question_by_name_query(db, decoded)
    if not question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question '{decoded}' not found",
        )
    return question


@router.post(
    "/",
    summary="Create a new question",
    description="Insert a new question. 400 if one with the same name exists.",
    response_model=QuestionSchema,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Question already exists or creation failed",
        },
    },
    dependencies=[api_key_dep],
    operation_id="createQuestion",
)
async def create_question(
    payload: CreateQuestionSchema,
    db: Database = db_dep,
) -> QuestionSchema:
    if await get_question_by_name_query(db, payload.name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Question '{payload.name}' already exists",
        )
    created = await create_question_query(db, payload)
    if not created:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create question",
        )
    return created


@router.put(
    "/{name:path}",
    summary="Update an existing question",
    description="Apply partial updates, 404 if not found.",
    response_model=QuestionSchema,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "No fields to update or update failed",
        },
        status.HTTP_404_NOT_FOUND: {"description": "Question not found"},
    },
    dependencies=[api_key_dep],
    operation_id="updateQuestion",
)
async def update_question(
    name: str,
    payload: UpdateQuestionSchema,
    db: Database = db_dep,
) -> QuestionSchema:
    decoded = urllib.parse.unquote(name)
    existing = await get_question_by_name_query(db, decoded)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question '{decoded}' not found",
        )
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No updates provided",
        )
    updated = await update_question_query(db, decoded, payload)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update question",
        )
    return updated


@router.delete(
    "/{name:path}",
    summary="Delete a question",
    description="Delete the question, and return the deleted record.",
    response_model=QuestionSchema,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Question not found"}},
    dependencies=[api_key_dep],
    operation_id="deleteQuestion",
)
async def delete_question(
    name: str,
    db: Database = db_dep,
) -> QuestionSchema:
    decoded = urllib.parse.unquote(name)
    existing = await get_question_by_name_query(db, decoded)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question '{decoded}' not found",
        )
    await delete_question_query(db, decoded)
    return existing


@router.get(
    "/nth/{n}",
    summary="Get the Nth question",
    description="Return the Nth question in insertion order (0-based), 404 if out of range.",
    response_model=QuestionSchema,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Index out of range"}},
    operation_id="getNthQuestion",
)
async def get_nth_question(
    n: int,
    db: Database = db_dep,
) -> QuestionSchema:
    question = await get_nth_question_query(db, n)
    if not question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No question at index {n}",
        )
    return question


@router.post(
    "/embed",
    summary="(Re)generate embeddings for all questions",
    description="Regenerates the embeddings for each question. Requires API key.",
    response_model=str,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[api_key_dep],
    operation_id="embedAllQuestions",
)
async def embed_questions(
    payload: EmbedQuestionsSchema,
    db: Database = db_dep,
) -> str:
    await fill_embeddings(db, payload.model, payload.all)
    return "Embeddings updated successfully"
