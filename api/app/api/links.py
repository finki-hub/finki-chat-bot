import urllib.parse

from fastapi import APIRouter, HTTPException

from app.constants import strings
from app.constants.errors import LINK_404
from app.data.links import (
    create_link_query,
    delete_link_query,
    get_link_by_name_query,
    get_link_names_query,
    get_links_query,
    get_nth_link_query,
    update_link_query,
)
from app.schema.link import CreateLinkSchema, LinkSchema, UpdateLinkSchema

router = APIRouter(tags=["Links"])


@router.get("/check", response_model=str)
async def check() -> str:
    return strings.API_RUNNING


@router.get("/list", response_model=list[LinkSchema])
async def get_links() -> list[LinkSchema]:
    result = await get_links_query()

    return result


@router.get("/names", response_model=list[str])
async def get_link_names() -> list[str]:
    result = await get_link_names_query()

    return result


@router.get("/name/{name:path}", response_model=LinkSchema)
async def get_link_by_name(name: str) -> LinkSchema:
    decoded_name = urllib.parse.unquote(name)

    result = await get_link_by_name_query(decoded_name)

    if not result:
        raise HTTPException(status_code=404, detail=LINK_404)

    return result


@router.post("/create", response_model=LinkSchema)
async def create_link(link: CreateLinkSchema) -> LinkSchema:
    existing_link = await get_link_by_name_query(link.name)

    if existing_link:
        raise HTTPException(status_code=400, detail="Link already exists")

    result = await create_link_query(link)

    if not result:
        raise HTTPException(status_code=400, detail="Failed to create link")

    return result


@router.put("/update/{name:path}", response_model=LinkSchema)
async def update_link(name: str, link: UpdateLinkSchema) -> LinkSchema:
    decoded_name = urllib.parse.unquote(name)
    existing_link = await get_link_by_name_query(decoded_name)

    if not existing_link:
        raise HTTPException(status_code=404, detail=LINK_404)

    updates = link.model_dump(exclude_unset=True)

    if len(updates) == 0:
        raise HTTPException(status_code=400, detail="No updates provided")

    result = await update_link_query(decoded_name, link)

    if not result:
        raise HTTPException(status_code=400, detail="Failed to update link")

    return result


@router.delete("/delete/{name:path}", response_model=LinkSchema)
async def delete_link(name: str) -> LinkSchema:
    decoded_name = urllib.parse.unquote(name)
    existing_link = await get_link_by_name_query(decoded_name)

    if not existing_link:
        raise HTTPException(status_code=404, detail=LINK_404)

    await delete_link_query(decoded_name)

    return existing_link


@router.get("/nth/{n}", response_model=LinkSchema)
async def get_nth_link(n: int) -> LinkSchema:
    result = await get_nth_link_query(n)

    if not result:
        raise HTTPException(status_code=404, detail=LINK_404)

    return result
