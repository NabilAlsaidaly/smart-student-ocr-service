import re


class PaymentReceiptOcrTextNormalizer:
    """
    Normalizes OCR text into parser-friendly text.

    This does not replace the raw OCR output. It only improves the public
    text field returned to Laravel so amount/reference parsing becomes more
    reliable.

    Examples:
    - 500D0OO -> 5000000
    - 5OOO -> 5000
    """

    _NUMERIC_LIKE_PATTERN = re.compile(
        r"(?<![A-Za-z])(?=[0-9OoDIl|.,]{3,})(?=[0-9OoDIl|.,]*[0-9])[0-9OoDIl|.,]+(?![A-Za-z])"
    )

    _NUMERIC_CHARACTER_MAP = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "D": "0",
            "I": "1",
            "l": "1",
            "|": "1",
        }
    )

    def normalize(
        self,
        text: str,
    ) -> str:
        normalized_lines = [
            self._normalize_line(line)
            for line in text.splitlines()
        ]

        return "\n".join(
            line
            for line in normalized_lines
            if line.strip() != ""
        )

    def _normalize_line(
        self,
        line: str,
    ) -> str:
        return self._NUMERIC_LIKE_PATTERN.sub(
            self._normalize_numeric_match,
            line,
        )

    def _normalize_numeric_match(
        self,
        match: re.Match[str],
    ) -> str:
        return match.group(0).translate(self._NUMERIC_CHARACTER_MAP)