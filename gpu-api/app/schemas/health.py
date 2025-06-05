from datetime import datetime

from pydantic import BaseModel, Field


class RootStatus(BaseModel):
    message: str = Field(
        examples=["gpu-api is running."],
        description="Simple liveness confirmation message",
    )


class DependencyStatus(BaseModel):
    status: str = Field(
        examples=["ok"],
        description="Low-level status string for this dependency",
    )
    healthy: bool = Field(
        examples=[True],
        description="True if this dependency is healthy, False otherwise",
    )


class HealthResponse(BaseModel):
    status: str = Field(
        examples=["ok"],
        description="Overall service health status",
    )
    timestamp: datetime = Field(
        examples=["2025-06-05T12:00:00Z"],
        description="UTC ISO-8601 timestamp of this health check",
    )
    dependencies: dict[str, DependencyStatus] = Field(
        ...,
        description="Mapping of each dependency name to its status",
    )
