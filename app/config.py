import os


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


settings = Settings()