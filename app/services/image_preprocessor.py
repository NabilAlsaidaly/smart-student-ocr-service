from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from app.exceptions import OcrServiceError


@dataclass(frozen=True)
class ImagePreprocessingResult:
    file_path: Path
    steps: list[str]


class ImagePreprocessor:
    """
    Conservative OpenCV preprocessing for receipt images.

    The goal is to improve readability without applying destructive transforms.
    More aggressive operations such as deskewing and document cropping will be
    introduced later after we test with real bank receipt samples.
    """

    def preprocess(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> ImagePreprocessingResult:
        image = self._decode_image(file_bytes)

        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised_image = cv2.fastNlMeansDenoising(
            grayscale_image,
            None,
            10,
            7,
            21,
        )
        enhanced_image = cv2.equalizeHist(denoised_image)

        output_file_path = self._write_processed_image(
            image=enhanced_image,
            filename=filename,
        )

        return ImagePreprocessingResult(
            file_path=output_file_path,
            steps=[
                "decode",
                "grayscale",
                "denoise",
                "contrast_equalization",
            ],
        )

    def _decode_image(self, file_bytes: bytes) -> np.ndarray:
        image_buffer = np.frombuffer(file_bytes, dtype=np.uint8)

        image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

        if image is None:
            raise OcrServiceError(
                message="Uploaded image could not be decoded.",
                status_code=422,
            )

        return image

    def _write_processed_image(
        self,
        image: np.ndarray,
        filename: str,
    ) -> Path:
        suffix = Path(filename).suffix.lower()

        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".png"

        with NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            output_file_path = Path(temporary_file.name)

        is_written = cv2.imwrite(str(output_file_path), image)

        if not is_written:
            output_file_path.unlink(missing_ok=True)

            raise OcrServiceError(
                message="Failed to write preprocessed image.",
                status_code=500,
            )

        return output_file_path