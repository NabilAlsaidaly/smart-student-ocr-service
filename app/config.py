import os


def env_bool(
    key: str,
    default: bool = False,
) -> bool:
    value = os.getenv(key)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class Settings:
    max_file_size_bytes: int = int(
        os.getenv("OCR_MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024))
    )

    allowed_mime_types: set[str] = {
        mime_type.strip()
        for mime_type in os.getenv(
            "OCR_ALLOWED_MIME_TYPES",
            "image/jpeg,image/png,image/webp,application/pdf",
        ).split(",")
        if mime_type.strip()
    }

    ocr_engine_driver: str = os.getenv(
        "OCR_ENGINE_DRIVER",
        "fake",
    ).strip().lower()

    easyocr_languages: list[str] = [
        language.strip()
        for language in os.getenv(
            "OCR_EASYOCR_LANGUAGES",
            "en,ar",
        ).split(",")
        if language.strip()
    ]

    easyocr_gpu: bool = env_bool(
        key="OCR_EASYOCR_GPU",
        default=False,
    )


settings = Settings()