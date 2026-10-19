"""Text and image field extraction for analyzed predictions.

Two layers:
1. ``read_image``             - OCR via Tesseract or EasyOCR.
2. ``parse_transaction_text`` - regex parsing of free text into structured fields.

A single transaction line can look like any of:
    "Amount 2350.00, Type TRANSFER, oldbal 12000, newbal 0"
    "amount=2350; type=TRANSFER; oldbalanceOrg=12000; newbalanceOrig=0"
    "2350 TRANSFER"   (unlabelled amount falls back to the largest number)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class OcrError(RuntimeError):
    """Raised when no OCR engine can process the input."""


class ImageTooLargeError(ValueError):
    """Raised when an upload exceeds the configured size limit."""


class UnsupportedImageError(ValueError):
    """Raised when the uploaded file is not a supported image."""


# --------------------------------------------------------------------------
# Text parsing
# --------------------------------------------------------------------------

_LABELLED_AMOUNT = re.compile(r"amount\s*[=:]\s*([\d.,]+)", re.IGNORECASE)
TYPE_LABEL_PATTERN = re.compile(r"type\s*[=:]\s*([A-Za-z_]+)", re.IGNORECASE)
_OLD_BALANCE_LABELLED = re.compile(r"old\s*balance\s*(?:org)?\s*[=:]\s*([\d.,]+)", re.IGNORECASE)
_OLD_BALANCE_SHORT = re.compile(r"oldbal(?:ance)?\s*[=: ]\s*([\d.,]+$|[\d.,]+)", re.IGNORECASE)
_NEW_BALANCE_LABELLED = re.compile(r"new\s*balance\s*(?:orig)?\s*[=:]\s*([\d.,]+)", re.IGNORECASE)
_NEW_BALANCE_SHORT = re.compile(r"newbal(?:ance)?\s*[=: ]\s*([\d.,]+$|[\d.,]+)", re.IGNORECASE)
_BARE_NUMBER = re.compile(r"\b\d[\d,]*\.?\d*\b")


def _currency(value: str) -> float | None:
    """Normalize a currency-ish string to a float."""
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _first(values) -> float | None:
    """First non-None value, or None."""
    return next((value for value in values if value is not None), None)


def _encode_type(raw: str) -> int | None:
    """Map a transaction type token onto the PaySim category encoding."""
    from fraudlens.etl.datasets import get_spec

    codes = get_spec("kaggle").category_codes
    return codes.get(str(raw).strip().upper().replace(" ", "_"))


def parse_transaction_text(text: str) -> dict[str, float | int | None]:
    """Extract Amount, Type and balance fields from free text.

    Amount resolution order:
      1. ``amount = 2350.00`` (labelled)
      2. the largest bare number in the text (unlabelled fallback)
    """
    lowered = (text or "").strip()
    labelled = _LABELLED_AMOUNT.findall(lowered)
    if labelled:
        amount = _first(_currency(value) for value in labelled)
    else:
        numbers = [value for value in (_currency(match) for match in _BARE_NUMBER.findall(lowered)) if value]
        amount = max(numbers) if numbers else None

    type_match = TYPE_LABEL_PATTERN.search(lowered)
    old_matches = _OLD_BALANCE_LABELLED.findall(lowered) or _OLD_BALANCE_SHORT.findall(lowered)
    new_matches = _NEW_BALANCE_LABELLED.findall(lowered) or _NEW_BALANCE_SHORT.findall(lowered)

    return {
        "amount": amount,
        "type_code": _encode_type(type_match.group(1)) if type_match else None,
        "oldbalance": _first(_currency(value) for value in old_matches),
        "newbalance": _first(_currency(value) for value in new_matches),
    }


def parse_old_balance(text: str) -> float | None:
    """Public helper: old balance from text, if present."""
    return _first(_currency(value) for value in _OLD_BALANCE_LABELLED.findall(text or ""))


def parse_new_balance(text: str) -> float | None:
    """Public helper: new balance from text, if present."""
    return _first(_currency(value) for value in _NEW_BALANCE_LABELLED.findall(text or ""))


# --------------------------------------------------------------------------
# Image OCR
# --------------------------------------------------------------------------


def validate_image(path: Path | str, max_bytes: int | None = None) -> Path:
    """Reject unsupported extensions and oversized uploads before OCR runs."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UnsupportedImageError(f"unsupported image extension: {path.suffix or '(none)'}")
    if not path.exists():
        raise UnsupportedImageError(f"image file not found: {path}")

    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise ImageTooLargeError(
            f"image is {path.stat().st_size} bytes; limit is {max_bytes} bytes"
        )
    return path


def read_image(path: Path | str) -> str:
    """Run OCR over an image using the first available engine."""
    raw_text, _engine = read_image_with_engine(path)
    return raw_text


def read_image_with_engine(path: Path | str) -> tuple[str, str]:
    """Run OCR over an image and return ``(recognized_text, engine_name)``."""
    path = validate_image(path)

    errors: dict[str, str] = {}
    for backend in ("tesseract", "easyocr"):
        if _engine_ready(backend):
            try:
                return _run_engine(backend, path), backend
            except Exception as exc:  # noqa: BLE001 - try the next engine
                errors[backend] = str(exc)

    raise OcrError(f"no OCR engine could process the image: {errors or 'none installed'}")


def _engine_ready(backend: str) -> bool:
    """True when the backend is installed and usable."""
    try:
        if backend == "tesseract":
            import pytesseract

            pytesseract.get_tesseract_version()
            return True
        if backend == "easyocr":
            import easyocr  # noqa: F401

            return True
    except Exception:  # noqa: BLE001 - missing binary or package
        return False
    return False


def _run_engine(backend: str, path: Path) -> str:
    if backend == "tesseract":
        import pytesseract
        from PIL import Image

        return pytesseract.image_to_string(Image.open(path))
    if backend == "easyocr":
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False)
        return "\n".join(reader.readtext(str(path), detail=0))
    raise OcrError(f"unsupported backend: {backend}")


def extract_fields_from_image(path: Path | str) -> dict[str, float | int | None]:
    """OCR an image, then parse the recognized text into structured fields."""
    return parse_transaction_text(read_image(path))
