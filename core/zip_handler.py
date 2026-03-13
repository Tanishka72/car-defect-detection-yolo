import zipfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def extract_zip(zip_path: str | Path, extract_to: str | Path) -> list[Path]:
    """Extract a ZIP file and return paths to all supported image files.

    Args:
        zip_path: Path to the ZIP file.
        extract_to: Directory to extract images into.

    Returns:
        Sorted list of Path objects for valid image files.

    Raises:
        FileNotFoundError: If zip_path does not exist.
        zipfile.BadZipFile: If the file is not a valid ZIP.
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Filter out directories and macOS resource forks
        members = [
            m for m in zf.namelist()
            if not m.startswith("__MACOSX") and not m.endswith("/")
        ]
        zf.extractall(extract_to, members=members)

    # Collect all image files recursively (ZIP may contain subfolders)
    image_files = sorted(
        p for p in extract_to.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    logger.info("Extracted %d image(s) from %s", len(image_files), zip_path.name)
    return image_files
