from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2

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

    The engine is rotation-aware because users may upload receipt photos taken
    sideways from mobile devices.
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
            candidates = self._read_rotation_candidates(
                prepared_file_path=prepared_file_path,
                mime_type=mime_type,
            )
        except OcrServiceError:
            raise
        except Exception as exception:
            raise OcrServiceError(
                message="EasyOCR failed to read the prepared receipt image.",
                status_code=500,
            ) from exception

        best_candidate = self._select_best_candidate(candidates)

        return PaymentReceiptOcrEngineResult(
            text=best_candidate["text"],
            confidence_score=best_candidate["confidence_score"],
            engine="easyocr",
            raw={
                "driver": "easyocr",
                "languages": list(self.languages),
                "gpu": bool(self.gpu),
                "mime_type": str(mime_type),
                "result_count": int(len(best_candidate["items"])),
                "orientation": {
                    "rotation_degrees": int(best_candidate["rotation_degrees"]),
                    "candidate_count": int(len(candidates)),
                    "score": float(best_candidate["score"]),
                },
                "items": best_candidate["items"],
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

    def _read_rotation_candidates(
        self,
        prepared_file_path: Path,
        mime_type: str,
    ) -> list[dict[str, Any]]:
        candidates = [
            self._read_candidate(
                file_path=prepared_file_path,
                rotation_degrees=0,
            )
        ]

        if not mime_type.startswith("image/"):
            return candidates

        rotated_file_paths = self._create_rotated_image_files(prepared_file_path)

        try:
            for rotation_degrees, rotated_file_path in rotated_file_paths:
                candidates.append(
                    self._read_candidate(
                        file_path=rotated_file_path,
                        rotation_degrees=rotation_degrees,
                    )
                )
        finally:
            for _, rotated_file_path in rotated_file_paths:
                rotated_file_path.unlink(missing_ok=True)

        return candidates

    def _read_candidate(
        self,
        file_path: Path,
        rotation_degrees: int,
    ) -> dict[str, Any]:
        results = self._reader().readtext(
            str(file_path),
            detail=1,
            paragraph=False,
        )

        normalized_results = self._normalize_results(results)
        text = "\n".join(self._group_text_lines(normalized_results))
        confidence_score = self._average_confidence(normalized_results)

        return {
            "rotation_degrees": rotation_degrees,
            "text": text,
            "confidence_score": confidence_score,
            "items": normalized_results,
            "score": self._score_candidate(
                text=text,
                confidence_score=confidence_score,
                result_count=len(normalized_results),
            ),
        }

    def _create_rotated_image_files(
        self,
        prepared_file_path: Path,
    ) -> list[tuple[int, Path]]:
        image = cv2.imread(str(prepared_file_path), cv2.IMREAD_COLOR)

        if image is None:
            return []

        rotation_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }

        rotated_file_paths: list[tuple[int, Path]] = []

        for rotation_degrees, rotation_code in rotation_map.items():
            rotated_image = cv2.rotate(image, rotation_code)
            rotated_file_path = self._write_rotation_candidate(
                image=rotated_image,
                rotation_degrees=rotation_degrees,
            )

            rotated_file_paths.append(
                (
                    rotation_degrees,
                    rotated_file_path,
                )
            )

        return rotated_file_paths

    def _write_rotation_candidate(
        self,
        image: Any,
        rotation_degrees: int,
    ) -> Path:
        with NamedTemporaryFile(
            delete=False,
            suffix=f".rot{rotation_degrees}.png",
        ) as temporary_file:
            output_file_path = Path(temporary_file.name)

        is_written = cv2.imwrite(
            str(output_file_path),
            image,
        )

        if not is_written:
            output_file_path.unlink(missing_ok=True)

            raise OcrServiceError(
                message="Failed to write rotated OCR image candidate.",
                status_code=500,
            )

        return output_file_path

    def _select_best_candidate(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return max(
            candidates,
            key=lambda candidate: candidate["score"],
        )

    def _score_candidate(
        self,
        text: str,
        confidence_score: float | None,
        result_count: int,
    ) -> float:
        normalized_text = text.lower()

        indicators = [
            "bank",
            "payment",
            "receipt",
            "amount",
            "transaction",
            "syp",
            "فاتورة",
            "حساب",
            "المبلغ",
            "ليرة",
            "سورية",
            "سوريا",
            "المصرف",
            "البنك",
            "رقم",
            "قسيمة",
            "ايصال",
            "إيصال",
            "سند",
            "دفع",
            "تحويل",
        ]

        indicator_hits = sum(
            1
            for indicator in indicators
            if indicator.lower() in normalized_text
        )

        has_digits = any(character.isdigit() for character in text)

        return (
            float(confidence_score or 0.0)
            + float(result_count * 2)
            + float(indicator_hits * 25)
            + (20.0 if has_digits else 0.0)
        )

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