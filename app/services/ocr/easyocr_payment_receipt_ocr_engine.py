from pathlib import Path
from typing import Any

from app.config import settings
from app.exceptions import OcrServiceError
from app.services.ocr.payment_receipt_ocr_engine import (
    PaymentReceiptOcrEngineResult,
)


class EasyOcrPaymentReceiptOcrEngine:
    """
    EasyOCR-based payment receipt OCR engine.

    This engine is loaded lazily so the service can still run and test with the
    fake engine when EasyOCR is not installed.
    """

    _reader_cache: dict[tuple[tuple[str, ...], bool], Any] = {}

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool | None = None,
        reader: Any | None = None,
    ) -> None:
        self.languages = languages or settings.easyocr_languages
        self.gpu = settings.easyocr_gpu if gpu is None else gpu
        self.reader = reader

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

        try:
            results = self._reader().readtext(
                str(prepared_file_path),
                detail=1,
                paragraph=False,
            )
        except OcrServiceError:
            raise
        except Exception as exception:
            raise OcrServiceError(
                message="EasyOCR failed to read the prepared receipt image.",
                status_code=500,
            ) from exception

        normalized_results = self._normalize_results(results)
        text_lines = [
            item["text"]
            for item in normalized_results
            if item["text"].strip() != ""
        ]

        return PaymentReceiptOcrEngineResult(
            text="\n".join(text_lines),
            confidence_score=self._average_confidence(normalized_results),
            engine="easyocr",
            raw={
                "driver": "easyocr",
                "languages": self.languages,
                "gpu": self.gpu,
                "mime_type": mime_type,
                "result_count": len(normalized_results),
                "items": normalized_results,
            },
        )

    def _reader(self) -> Any:
        if self.reader is not None:
            return self.reader

        cache_key = (
            tuple(self.languages),
            self.gpu,
        )

        if cache_key not in self._reader_cache:
            try:
                import easyocr
            except ImportError as exception:
                raise OcrServiceError(
                    message="EasyOCR is not installed. Install optional EasyOCR dependencies first.",
                    status_code=500,
                ) from exception

            self._reader_cache[cache_key] = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
            )

        return self._reader_cache[cache_key]

    def _normalize_results(
        self,
        results: list[Any],
    ) -> list[dict[str, Any]]:
        normalized_results: list[dict[str, Any]] = []

        for result in results:
            if not isinstance(result, (list, tuple)) or len(result) < 3:
                continue

            bbox = result[0]
            text = str(result[1] or "")
            confidence = self._normalize_confidence(result[2])
            left, top = self._bbox_top_left(bbox)

            normalized_results.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox,
                    "left": left,
                    "top": top,
                }
            )

        return sorted(
            normalized_results,
            key=lambda item: (
                item["top"],
                item["left"],
            ),
        )

    def _normalize_confidence(self, confidence: Any) -> float | None:
        try:
            normalized_confidence = float(confidence)
        except (TypeError, ValueError):
            return None

        if normalized_confidence <= 1:
            normalized_confidence *= 100

        normalized_confidence = max(
            0.0,
            min(100.0, normalized_confidence),
        )

        return round(normalized_confidence, 2)

    def _average_confidence(
        self,
        normalized_results: list[dict[str, Any]],
    ) -> float | None:
        confidence_values = [
            item["confidence"]
            for item in normalized_results
            if item["confidence"] is not None
        ]

        if not confidence_values:
            return None

        return round(
            sum(confidence_values) / len(confidence_values),
            2,
        )

    def _bbox_top_left(
        self,
        bbox: Any,
    ) -> tuple[float, float]:
        try:
            points = list(bbox)
            x_values = [float(point[0]) for point in points]
            y_values = [float(point[1]) for point in points]

            return min(x_values), min(y_values)
        except (TypeError, ValueError, IndexError):
            return 0.0, 0.0