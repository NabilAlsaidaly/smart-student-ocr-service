from typing import Any

from pydantic import BaseModel, Field


class PaymentReceiptOcrResponse(BaseModel):
    text: str
    confidence_score: float | None = Field(default=None)
    engine: str
    raw: dict[str, Any] = Field(default_factory=dict)