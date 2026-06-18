import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


client = TestClient(app)


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


def test_payment_receipt_ocr_endpoint_returns_fake_ocr_result_for_valid_image() -> None:
    response = client.post(
        "/api/ocr/payment-receipts/read",
        files={
            "file": (
                "receipt.png",
                make_valid_png_bytes(),
                "image/png",
            ),
        },
        data={
            "mime_type": "image/png",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["text"] == "\n".join(
        [
            "BANK PAYMENT RECEIPT",
            "Bank: Python OCR Bank",
            "Receipt No: OCR-HTTP-001",
            "Transaction No: TX-HTTP-001",
            "Amount: 5000000",
            "Currency: SYP",
        ]
    )
    assert payload["confidence_score"] == 91.75
    assert payload["engine"] == "python-fake-ocr"
    assert payload["raw"]["filename"] == "receipt.png"
    assert payload["raw"]["mime_type"] == "image/png"
    assert payload["raw"]["preprocessing"] == [
        "decode",
        "grayscale",
        "denoise",
        "contrast_equalization",
    ]
    assert payload["raw"]["engine"]["driver"] == "fake"
    assert payload["raw"]["engine"]["mime_type"] == "image/png"
    assert payload["raw"]["engine"]["prepared_file_suffix"] == ".png"

def test_payment_receipt_ocr_endpoint_rejects_invalid_image_content() -> None:
    response = client.post(
        "/api/ocr/payment-receipts/read",
        files={
            "file": (
                "receipt.png",
                b"this is not a real image",
                "image/png",
            ),
        },
        data={
            "mime_type": "image/png",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Uploaded image could not be decoded.",
    }


def test_payment_receipt_ocr_endpoint_rejects_unsupported_mime_type() -> None:
    response = client.post(
        "/api/ocr/payment-receipts/read",
        files={
            "file": (
                "receipt.txt",
                b"fake text content",
                "text/plain",
            ),
        },
        data={
            "mime_type": "text/plain",
        },
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "Unsupported file MIME type [text/plain].",
    }


def test_payment_receipt_ocr_endpoint_rejects_empty_file() -> None:
    response = client.post(
        "/api/ocr/payment-receipts/read",
        files={
            "file": (
                "receipt.png",
                b"",
                "image/png",
            ),
        },
        data={
            "mime_type": "image/png",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Uploaded file is empty.",
    }


def test_payment_receipt_ocr_endpoint_rejects_large_file() -> None:
    original_max_file_size = settings.max_file_size_bytes

    settings.max_file_size_bytes = 5

    try:
        response = client.post(
            "/api/ocr/payment-receipts/read",
            files={
                "file": (
                    "receipt.png",
                    b"this content is larger than five bytes",
                    "image/png",
                ),
            },
            data={
                "mime_type": "image/png",
            },
        )
    finally:
        settings.max_file_size_bytes = original_max_file_size

    assert response.status_code == 413
    assert response.json() == {
        "detail": "File too large.",
    }


def test_payment_receipt_ocr_endpoint_rejects_blank_filename() -> None:
    response = client.post(
        "/api/ocr/payment-receipts/read",
        files={
            "file": (
                " ",
                make_valid_png_bytes(),
                "image/png",
            ),
        },
        data={
            "mime_type": "image/png",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Uploaded file must have a filename.",
    }