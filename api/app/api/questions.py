import urllib

from fastapi import APIRouter, HTTPException

from app.constants import strings
from app.data.questions import (
    create_question_query,
    delete_question_query,
    get_nth_question_query,
    get_question_by_name_query,
    get_question_names_query,
    get_questions_query,
    update_question_query,
)
from app.schema.question import (
    CreateQuestionSchema,
    QuestionSchema,
    UpdateQuestionSchema,
)

router = APIRouter(tags=["Questions"])


@router.get("/check", response_model=str)
async def check() -> str:
    return strings.alive_response


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
        raise HTTPException(status_code=404, detail="Question not found")

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


@router.put("/update/{name}", response_model=QuestionSchema)
async def update_question(name: str, question: UpdateQuestionSchema) -> QuestionSchema:
    decoded_name = urllib.parse.unquote(name)
    existing_question = await get_question_by_name_query(decoded_name)

    if not existing_question:
        raise HTTPException(status_code=404, detail="Question not found")

    updates = question.model_dump(exclude_unset=True)

    if len(updates) == 0:
        raise HTTPException(status_code=400, detail="No updates provided")

    result = await update_question_query(decoded_name, question)

    if not result:
        raise HTTPException(status_code=400, detail="Failed to update question")

    return result


@router.delete("/delete/{name}", response_model=QuestionSchema)
async def delete_question(name: str) -> QuestionSchema:
    decoded_name = urllib.parse.unquote(name)
    existing_question = await get_question_by_name_query(decoded_name)

    if not existing_question:
        raise HTTPException(status_code=404, detail="Question not found")

    await delete_question_query(decoded_name)

    return existing_question


@router.get("/nth/{n}", response_model=QuestionSchema)
async def get_nth_question(n: int) -> QuestionSchema:
    result = await get_nth_question_query(n)

    if not result:
        raise HTTPException(status_code=404, detail="Question not found")

    return result
