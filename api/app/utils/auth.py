from fastapi import HTTPException, Request


def verify_api_key(request: Request) -> None:
    """
    Verify the API key from the request headers against the configured API key.
    Raises HTTPException with status code 401 if the API key is invalid or missing.
    """
    key = request.headers.get("x-api-key")
    api_key = request.app.state.settings.API_KEY

    if key != api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key",
        )
