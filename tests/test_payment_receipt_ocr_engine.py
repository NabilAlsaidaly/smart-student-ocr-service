from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from app.exceptions import OcrServiceError
from app.services.ocr.payment_receipt_ocr_engine import FakePaymentReceiptOcrEngine


def test_fake_payment_receipt_ocr_engine_returns_deterministic_result() -> None:
    with NamedTemporaryFile(delete=False, suffix=".png") as temporary_file:
        temporary_file.write(b"prepared image content")
        temporary_file_path = Path(temporary_file.name)

    try:
        result = FakePaymentReceiptOcrEngine().read(
            prepared_file_path=temporary_file_path,
            mime_type="image/png",
        )

        assert result.text == "\n".join(
            [
                "BANK PAYMENT RECEIPT",
                "Bank: Python OCR Bank",
                "Receipt No: OCR-HTTP-001",
                "Transaction No: TX-HTTP-001",
                "Amount: 5000000",
                "Currency: SYP",
            ]
        )
        assert result.confidence_score == 91.75
        assert result.engine == "python-fake-ocr"
        assert result.raw == {
            "driver": "fake",
            "mime_type": "image/png",
            "prepared_file_suffix": ".png",
        }
    finally:
        temporary_file_path.unlink(missing_ok=True)


def test_fake_payment_receipt_ocr_engine_rejects_missing_prepared_file() -> None:
    with pytest.raises(OcrServiceError) as exception_info:
        FakePaymentReceiptOcrEngine().read(
            prepared_file_path=Path("missing-prepared-file.png"),
            mime_type="image/png",
        )

    assert exception_info.value.message == "Prepared OCR file does not exist."
    assert exception_info.value.status_code == 500