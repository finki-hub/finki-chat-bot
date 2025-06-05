from fastapi import Request

from app.data.connection import Database


def get_db(request: Request) -> Database:
    """
    Dependency to get the database connection from the request's app state.
    """
    return request.app.state.db
