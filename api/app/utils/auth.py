from fastapi import HTTPException, Request


def verify_api_key(request: Request) -> None:
    key = request.headers.get("x-api-key")
    api_key = request.app.state.settings.API_KEY

    if key != api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key",
        )
