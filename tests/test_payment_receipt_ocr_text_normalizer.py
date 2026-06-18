from app.services.ocr.payment_receipt_ocr_text_normalizer import (
    PaymentReceiptOcrTextNormalizer,
)


def test_payment_receipt_ocr_text_normalizer_corrects_numeric_ocr_confusions() -> None:
    normalized_text = PaymentReceiptOcrTextNormalizer().normalize(
        "Amount 500D0OO SYP"
    )

    assert normalized_text == "Amount 5000000 SYP"


def test_payment_receipt_ocr_text_normalizer_does_not_change_reference_numbers() -> None:
    normalized_text = PaymentReceiptOcrTextNormalizer().normalize(
        "Receipt No OCR-TEST-001"
    )

    assert normalized_text == "Receipt No OCR-TEST-001"


def test_payment_receipt_ocr_text_normalizer_preserves_regular_text() -> None:
    normalized_text = PaymentReceiptOcrTextNormalizer().normalize(
        "BANK PAYMENT RECEIPT"
    )

    assert normalized_text == "BANK PAYMENT RECEIPT"


def test_payment_receipt_ocr_text_normalizer_removes_blank_lines() -> None:
    normalized_text = PaymentReceiptOcrTextNormalizer().normalize(
        "BANK\n\nAmount 500D0OO SYP\n"
    )

    assert normalized_text == "BANK\nAmount 5000000 SYP"