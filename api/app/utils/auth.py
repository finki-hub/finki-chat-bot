from fastapi import HTTPException, Request

from app.utils.config import API_KEY


def verify_api_key(request: Request) -> None:
    key = request.headers.get("x-api-key")

    if key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key",
        )
