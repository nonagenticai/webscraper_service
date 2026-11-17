"""Health check response schemas.

This module defines Pydantic models for health check API responses.
"""

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check response model for the /health endpoint.

    Attributes:
        status: Service health status ('healthy' or 'unhealthy').
        timestamp: ISO8601 formatted timestamp of the check.
        service: Name of the service.
        version: Semantic version of the service.
        playwright_available: Whether Playwright library is installed.
        browser_ready: Whether browser instance is initialized.
    """

    status: str = Field(description="Service health status")
    timestamp: str = Field(description="ISO8601 timestamp")
    service: str = Field(default="webscraper-service", description="Service name")
    version: str = Field(default="1.0.0", description="Service version")
    playwright_available: bool = Field(default=False, description="Playwright browser available")
    browser_ready: bool = Field(default=False, description="Browser initialized and ready")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00Z",
                "service": "webscraper-service",
                "version": "1.0.0",
                "playwright_available": True,
                "browser_ready": True,
            }
        }
    )
