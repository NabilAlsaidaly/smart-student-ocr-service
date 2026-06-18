from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from app.exceptions import OcrServiceError
from app.services.ocr.easyocr_payment_receipt_ocr_engine import (
    EasyOcrPaymentReceiptOcrEngine,
)


class FakeEasyOcrReader:
    def readtext(
        self,
        image_path: str,
        detail: int,
        paragraph: bool,
    ) -> list[tuple[list[list[np.int32]], str, float]]:
        return [
            (
                [
                    [np.int32(0), np.int32(40)],
                    [np.int32(100), np.int32(40)],
                    [np.int32(100), np.int32(60)],
                    [np.int32(0), np.int32(60)],
                ],
                "Amount 5000000 SYP",
                np.float32(0.80),
            ),
            (
                [
                    [np.int32(0), np.int32(10)],
                    [np.int32(100), np.int32(10)],
                    [np.int32(100), np.int32(30)],
                    [np.int32(0), np.int32(30)],
                ],
                "Bank: Test Bank",
                np.float32(0.90),
            ),
        ]


class EmptyEasyOcrReader:
    def readtext(
        self,
        image_path: str,
        detail: int,
        paragraph: bool,
    ) -> list:
        return []


class FailingEasyOcrReader:
    def readtext(
        self,
        image_path: str,
        detail: int,
        paragraph: bool,
    ) -> list:
        raise RuntimeError("OCR failed.")


def test_easyocr_payment_receipt_ocr_engine_returns_ordered_text_and_average_confidence() -> None:
    with NamedTemporaryFile(delete=False, suffix=".png") as temporary_file:
        temporary_file.write(b"prepared image content")
        temporary_file_path = Path(temporary_file.name)

    try:
        result = EasyOcrPaymentReceiptOcrEngine(
            languages=["en", "ar"],
            gpu=False,
            reader=FakeEasyOcrReader(),
        ).read(
            prepared_file_path=temporary_file_path,
            mime_type="image/png",
        )

        assert result.text == "\n".join(
            [
                "Bank: Test Bank",
                "Amount 5000000 SYP",
            ]
        )
        assert result.confidence_score == 85.0
        assert result.engine == "easyocr"
        assert result.raw["driver"] == "easyocr"
        assert result.raw["languages"] == ["en", "ar"]
        assert result.raw["gpu"] is False
        assert result.raw["mime_type"] == "image/png"
        assert result.raw["result_count"] == 2

        first_item = result.raw["items"][0]

        assert first_item["bbox"] == [
            [0.0, 10.0],
            [100.0, 10.0],
            [100.0, 30.0],
            [0.0, 30.0],
        ]
        assert isinstance(first_item["left"], float)
        assert isinstance(first_item["top"], float)
    finally:
        temporary_file_path.unlink(missing_ok=True)


def test_easyocr_payment_receipt_ocr_engine_returns_empty_text_when_no_results_are_found() -> None:
    with NamedTemporaryFile(delete=False, suffix=".png") as temporary_file:
        temporary_file.write(b"prepared image content")
        temporary_file_path = Path(temporary_file.name)

    try:
        result = EasyOcrPaymentReceiptOcrEngine(
            languages=["en", "ar"],
            gpu=False,
            reader=EmptyEasyOcrReader(),
        ).read(
            prepared_file_path=temporary_file_path,
            mime_type="image/png",
        )

        assert result.text == ""
        assert result.confidence_score is None
        assert result.engine == "easyocr"
        assert result.raw["result_count"] == 0
    finally:
        temporary_file_path.unlink(missing_ok=True)


def test_easyocr_payment_receipt_ocr_engine_rejects_missing_prepared_file() -> None:
    with pytest.raises(OcrServiceError) as exception_info:
        EasyOcrPaymentReceiptOcrEngine(
            languages=["en", "ar"],
            gpu=False,
            reader=FakeEasyOcrReader(),
        ).read(
            prepared_file_path=Path("missing-prepared-file.png"),
            mime_type="image/png",
        )

    assert exception_info.value.message == "Prepared OCR file does not exist."
    assert exception_info.value.status_code == 500


def test_easyocr_payment_receipt_ocr_engine_wraps_reader_failures() -> None:
    with NamedTemporaryFile(delete=False, suffix=".png") as temporary_file:
        temporary_file.write(b"prepared image content")
        temporary_file_path = Path(temporary_file.name)

    try:
        with pytest.raises(OcrServiceError) as exception_info:
            EasyOcrPaymentReceiptOcrEngine(
                languages=["en", "ar"],
                gpu=False,
                reader=FailingEasyOcrReader(),
            ).read(
                prepared_file_path=temporary_file_path,
                mime_type="image/png",
            )
    finally:
        temporary_file_path.unlink(missing_ok=True)

    assert exception_info.value.message == "EasyOCR failed to read the prepared receipt image."
    assert exception_info.value.status_code == 500