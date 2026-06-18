from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.config import settings
from app.exceptions import OcrServiceError
from app.schemas import PaymentReceiptOcrResponse
from app.services.payment_receipt_ocr_service import PaymentReceiptOcrService

app = FastAPI(
    title="Smart Student OCR Service",
    version="0.1.0",
)

payment_receipt_ocr_service = PaymentReceiptOcrService()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "ocr-service",
    }


@app.post(
    "/api/ocr/payment-receipts/read",
    response_model=PaymentReceiptOcrResponse,
)
async def read_payment_receipt(
    file: UploadFile = File(...),
    mime_type: str | None = Form(default=None),
) -> PaymentReceiptOcrResponse:
    if file.filename is None or file.filename.strip() == "":
        raise HTTPException(
            status_code=422,
            detail="Uploaded file must have a filename.",
        )

    contents = await file.read()

    if len(contents) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail="File too large.",
        )

    await file.seek(0)

    effective_mime_type = mime_type or file.content_type

    try:
        return payment_receipt_ocr_service.read_payment_receipt(
            file_bytes=contents,
            filename=file.filename,
            mime_type=effective_mime_type,
        )
    except OcrServiceError as exception:
        raise HTTPException(
            status_code=exception.status_code,
            detail=exception.message,
        ) from exception