from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.exceptions import OcrServiceError


@dataclass(frozen=True)
class PaymentReceiptOcrEngineResult:
    text: str
    confidence_score: float | None
    engine: str
    raw: dict[str, Any]


class PaymentReceiptOcrEngine(Protocol):
    def read(
        self,
        prepared_file_path: Path,
        mime_type: str,
    ) -> PaymentReceiptOcrEngineResult:
        ...


class FakePaymentReceiptOcrEngine:
    """
    Deterministic fake OCR engine.

    This keeps the OCR service testable while the real OCR engine is not installed yet.
    Later, EasyOCR/PaddleOCR will implement the same PaymentReceiptOcrEngine protocol.
    """

    def read(
        self,
        prepared_file_path: Path,
        mime_type: str,
    ) -> PaymentReceiptOcrEngineResult:
        if not prepared_file_path.is_file():
            raise OcrServiceError(
                message="Prepared OCR file does not exist.",
                status_code=500,
            )

        return PaymentReceiptOcrEngineResult(
            text="\n".join(
                [
                    "BANK PAYMENT RECEIPT",
                    "Bank: Python OCR Bank",
                    "Receipt No: OCR-HTTP-001",
                    "Transaction No: TX-HTTP-001",
                    "Amount: 5000000",
                    "Currency: SYP",
                ]
            ),
            confidence_score=91.75,
            engine="python-fake-ocr",
            raw={
                "driver": "fake",
                "mime_type": mime_type,
                "prepared_file_suffix": prepared_file_path.suffix,
            },
        )