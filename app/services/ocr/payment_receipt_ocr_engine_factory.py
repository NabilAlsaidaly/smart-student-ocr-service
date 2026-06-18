from app.config import settings
from app.exceptions import OcrServiceError
from app.services.ocr.easyocr_payment_receipt_ocr_engine import (
    EasyOcrPaymentReceiptOcrEngine,
)
from app.services.ocr.payment_receipt_ocr_engine import (
    FakePaymentReceiptOcrEngine,
    PaymentReceiptOcrEngine,
)


class PaymentReceiptOcrEngineFactory:
    def make(self) -> PaymentReceiptOcrEngine:
        driver = settings.ocr_engine_driver.strip().lower()

        if driver == "fake":
            return FakePaymentReceiptOcrEngine()

        if driver == "easyocr":
            return EasyOcrPaymentReceiptOcrEngine()

        raise OcrServiceError(
            message=f"Unsupported OCR engine driver [{driver}].",
            status_code=500,
        )