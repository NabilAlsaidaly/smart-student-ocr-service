import pytest

from app.config import settings
from app.exceptions import OcrServiceError
from app.services.ocr.easyocr_payment_receipt_ocr_engine import (
    EasyOcrPaymentReceiptOcrEngine,
)
from app.services.ocr.payment_receipt_ocr_engine import FakePaymentReceiptOcrEngine
from app.services.ocr.payment_receipt_ocr_engine_factory import (
    PaymentReceiptOcrEngineFactory,
)


def test_payment_receipt_ocr_engine_factory_returns_fake_engine() -> None:
    original_driver = settings.ocr_engine_driver

    settings.ocr_engine_driver = "fake"

    try:
        engine = PaymentReceiptOcrEngineFactory().make()
    finally:
        settings.ocr_engine_driver = original_driver

    assert isinstance(engine, FakePaymentReceiptOcrEngine)


def test_payment_receipt_ocr_engine_factory_returns_easyocr_engine() -> None:
    original_driver = settings.ocr_engine_driver

    settings.ocr_engine_driver = "easyocr"

    try:
        engine = PaymentReceiptOcrEngineFactory().make()
    finally:
        settings.ocr_engine_driver = original_driver

    assert isinstance(engine, EasyOcrPaymentReceiptOcrEngine)


def test_payment_receipt_ocr_engine_factory_rejects_unsupported_driver() -> None:
    original_driver = settings.ocr_engine_driver

    settings.ocr_engine_driver = "unsupported"

    try:
        with pytest.raises(OcrServiceError) as exception_info:
            PaymentReceiptOcrEngineFactory().make()
    finally:
        settings.ocr_engine_driver = original_driver

    assert exception_info.value.message == "Unsupported OCR engine driver [unsupported]."
    assert exception_info.value.status_code == 500