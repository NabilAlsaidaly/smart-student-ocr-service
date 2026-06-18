# Smart Student OCR Service

Independent FastAPI OCR service for payment receipt processing.

## Current status

This service currently returns deterministic fake OCR results through HTTP.

Laravel communicates with this service through a server-to-server HTTP request.

## Next steps

- Add Python tests.
- Add OpenCV preprocessing.
- Add EasyOCR or PaddleOCR.
- Return real extracted text and confidence score.

## Environment variables

```env
OCR_MAX_FILE_SIZE_BYTES=10485760
OCR_ALLOWED_MIME_TYPES=image/jpeg,image/png,image/webp,application/pdf
```

## Run locally

```bash
cd C:\laravel_project\smart-student-ocr-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

## Health check

```text
http://127.0.0.1:8001/health
```

## OCR endpoint

```text
POST /api/ocr/payment-receipts/read
multipart/form-data:
- file
- mime_type
```
