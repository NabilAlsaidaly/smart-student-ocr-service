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

    EasyOCR may return numpy scalar values inside bounding boxes. Before
    returning the OCR result, all values must be normalized into JSON-safe
    Python native types.
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

        return PaymentReceiptOcrEngineResult(
            text="\n".join(self._group_text_lines(normalized_results)),
            confidence_score=self._average_confidence(normalized_results),
            engine="easyocr",
            raw={
                "driver": "easyocr",
                "languages": list(self.languages),
                "gpu": bool(self.gpu),
                "mime_type": str(mime_type),
                "result_count": int(len(normalized_results)),
                "items": normalized_results,
            },
        )

    def _reader(self) -> Any:
        if self.reader is not None:
            return self.reader

        cache_key = (
            tuple(self.languages),
            bool(self.gpu),
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

            raw_bbox = result[0]
            text = str(result[1] or "")
            confidence = self._normalize_confidence(result[2])
            bbox = self._normalize_bbox(raw_bbox)
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

    def _group_text_lines(
        self,
        normalized_results: list[dict[str, Any]],
    ) -> list[str]:
        line_threshold = 20.0
        grouped_lines: list[list[dict[str, Any]]] = []

        for item in normalized_results:
            if str(item["text"]).strip() == "":
                continue

            matching_line = self._find_matching_line(
                grouped_lines=grouped_lines,
                item=item,
                line_threshold=line_threshold,
            )

            if matching_line is None:
                grouped_lines.append([item])
            else:
                matching_line.append(item)

        lines: list[str] = []

        for line_items in grouped_lines:
            sorted_line_items = sorted(
                line_items,
                key=lambda item: item["left"],
            )

            lines.append(
                " ".join(
                    str(item["text"]).strip()
                    for item in sorted_line_items
                    if str(item["text"]).strip() != ""
                )
            )

        return lines

    def _find_matching_line(
        self,
        grouped_lines: list[list[dict[str, Any]]],
        item: dict[str, Any],
        line_threshold: float,
    ) -> list[dict[str, Any]] | None:
        for line_items in grouped_lines:
            reference_top = float(line_items[0]["top"])
            item_top = float(item["top"])

            if abs(reference_top - item_top) <= line_threshold:
                return line_items

        return None

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

    def _normalize_bbox(self, bbox: Any) -> list[list[float]]:
        normalized_points: list[list[float]] = []

        try:
            for point in bbox:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue

                normalized_points.append(
                    [
                        float(point[0]),
                        float(point[1]),
                    ]
                )
        except TypeError:
            return []

        return normalized_points

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
        bbox: list[list[float]],
    ) -> tuple[float, float]:
        if not bbox:
            return 0.0, 0.0

        try:
            x_values = [float(point[0]) for point in bbox]
            y_values = [float(point[1]) for point in bbox]

            return min(x_values), min(y_values)
        except (TypeError, ValueError, IndexError):
            return 0.0, 0.0