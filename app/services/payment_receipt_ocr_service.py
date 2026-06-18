from pathlib import Path
from tempfile import NamedTemporaryFile

from app.config import settings
from app.exceptions import OcrServiceError
from app.schemas import PaymentReceiptOcrResponse


class PaymentReceiptOcrService:
    """
    Temporary OCR service implementation.

    This service currently returns deterministic fake OCR text.
    Later, OpenCV preprocessing and EasyOCR/PaddleOCR will be added here
    without changing the HTTP contract with Laravel.
    """

    def read_payment_receipt(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str | None,
    ) -> PaymentReceiptOcrResponse:
        self._validate_file_content(file_bytes)
        self._validate_mime_type(mime_type)

        temporary_file_path = self._store_temporary_file(
            file_bytes=file_bytes,
            filename=filename,
        )

        try:
            return PaymentReceiptOcrResponse(
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
                    "filename": filename,
                    "mime_type": mime_type,
                    "temporary_file": str(temporary_file_path),
                    "preprocessing": [],
                },
            )
        finally:
            temporary_file_path.unlink(missing_ok=True)

    def _validate_file_content(self, file_bytes: bytes) -> None:
        if len(file_bytes) == 0:
            raise OcrServiceError(
                message="Uploaded file is empty.",
                status_code=422,
            )

    def _validate_mime_type(self, mime_type: str | None) -> None:
        if mime_type is None or mime_type.strip() == "":
            raise OcrServiceError(
                message="Uploaded file MIME type is missing.",
                status_code=422,
            )

        normalized_mime_type = mime_type.strip().lower()

        if normalized_mime_type not in settings.allowed_mime_types:
            raise OcrServiceError(
                message=f"Unsupported file MIME type [{normalized_mime_type}].",
                status_code=415,
            )

    def _store_temporary_file(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> Path:
        suffix = Path(filename).suffix or ".bin"

        with NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(file_bytes)

            return Path(temporary_file.name)