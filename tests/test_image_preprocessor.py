import cv2
import numpy as np
import pytest

from app.exceptions import OcrServiceError
from app.services.image_preprocessor import ImagePreprocessor


def make_valid_png_bytes() -> bytes:
    image = np.full(
        shape=(64, 64, 3),
        fill_value=255,
        dtype=np.uint8,
    )

    cv2.rectangle(
        img=image,
        pt1=(8, 8),
        pt2=(56, 56),
        color=(0, 0, 0),
        thickness=2,
    )

    is_encoded, encoded_image = cv2.imencode(".png", image)

    assert is_encoded is True

    return encoded_image.tobytes()


def test_image_preprocessor_creates_processed_image_file() -> None:
    result = ImagePreprocessor().preprocess(
        file_bytes=make_valid_png_bytes(),
        filename="receipt.png",
    )

    try:
        assert result.file_path.is_file()
        assert result.steps == [
            "decode",
            "grayscale",
            "denoise",
            "contrast_equalization",
        ]
    finally:
        result.file_path.unlink(missing_ok=True)


def test_image_preprocessor_rejects_invalid_image_content() -> None:
    with pytest.raises(OcrServiceError) as exception_info:
        ImagePreprocessor().preprocess(
            file_bytes=b"not a real image",
            filename="receipt.png",
        )

    assert exception_info.value.message == "Uploaded image could not be decoded."
    assert exception_info.value.status_code == 422