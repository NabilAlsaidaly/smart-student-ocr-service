from pathlib import Path
from tempfile import NamedTemporaryFile

from app.config import settings
from app.exceptions import OcrServiceError
from app.schemas import PaymentReceiptOcrResponse
from app.services.image_preprocessor import ImagePreprocessor


class PaymentReceiptOcrService:
    """
    Temporary OCR service implementation.

    This service now performs conservative OpenCV preprocessing for image files,
    then returns deterministic fake OCR text.

    In the next OCR phase, EasyOCR/PaddleOCR will consume the prepared file
    without changing the HTTP contract with Laravel.
    """

    def __init__(
        self,
        image_preprocessor: ImagePreprocessor | None = None,
    ) -> None:
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()

    def read_payment_receipt(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str | None,
    ) -> PaymentReceiptOcrResponse:
        self._validate_file_content(file_bytes)
        normalized_mime_type = self._validate_mime_type(mime_type)

        prepared_file_path: Path | None = None
        preprocessing_steps: list[str] = []

        try:
            if self._is_image_mime_type(normalized_mime_type):
                preprocessing_result = self.image_preprocessor.preprocess(
                    file_bytes=file_bytes,
                    filename=filename,
                )

                prepared_file_path = preprocessing_result.file_path
                preprocessing_steps = preprocessing_result.steps
            else:
                prepared_file_path = self._store_temporary_file(
                    file_bytes=file_bytes,
                    filename=filename,
                )

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
                    "mime_type": normalized_mime_type,
                    "prepared_file": str(prepared_file_path),
                    "preprocessing": preprocessing_steps,
                },
            )
        finally:
            if prepared_file_path is not None:
                prepared_file_path.unlink(missing_ok=True)

    def _validate_file_content(self, file_bytes: bytes) -> None:
        if len(file_bytes) == 0:
            raise OcrServiceError(
                message="Uploaded file is empty.",
                status_code=422,
            )

    def _validate_mime_type(self, mime_type: str | None) -> str:
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

        return normalized_mime_type

    def _is_image_mime_type(self, mime_type: str) -> bool:
        return mime_type.startswith("image/")

    def _store_temporary_file(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> Path:
        suffix = Path(filename).suffix or ".bin"

        with NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(file_bytes)

            return Path(temporary_file.name)