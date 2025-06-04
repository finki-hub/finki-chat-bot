from fastapi import Request

from app.data.connection import Database


def get_db(request: Request) -> Database:
    return request.app.state.db
