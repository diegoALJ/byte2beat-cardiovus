"""Download the KCNH2 MaveDB score set and associated metadata.

Outputs
-------
data/raw/mavedb/score_set_metadata.json
data/raw/mavedb/scores.csv
data/raw/mavedb/mapped_variants.json

Usage
-----
From the repository root:

    python src/cardiovus/data/download_mavedb.py

To overwrite existing files:

    python src/cardiovus/data/download_mavedb.py --force
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Literal

import requests


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

API_BASE_URL = "https://api.mavedb.org/api/v1"

SCORE_SET_URN = "urn:mavedb:00001231-a-2"

# File location:
# repository/src/cardiovus/data/download_mavedb.py
# parents[3] therefore points to the repository root.
ROOT_DIR = Path(__file__).resolve().parents[3]

OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "mavedb"

ENDPOINTS = {
    "score_set_metadata.json": (
        f"{API_BASE_URL}/score-sets/{SCORE_SET_URN}",
        "json",
    ),
    "scores.csv": (
        f"{API_BASE_URL}/score-sets/{SCORE_SET_URN}/scores",
        "csv",
    ),
    "mapped_variants.json": (
        f"{API_BASE_URL}/score-sets/{SCORE_SET_URN}/mapped-variants",
        "json",
    ),
}

FileFormat = Literal["json", "csv"]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def validate_json(file_path: Path) -> None:
    """Verify that a downloaded file contains valid JSON."""
    try:
        with file_path.open("r", encoding="utf-8") as file:
            json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(
            f"Downloaded file is not valid JSON: {file_path}"
        ) from error


def validate_csv(file_path: Path) -> None:
    """Perform a basic validation of the downloaded CSV file."""
    try:
        with file_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.reader(file)
            header = next(reader, None)

    except UnicodeDecodeError as error:
        raise ValueError(
            f"Downloaded file is not valid UTF-8 CSV: {file_path}"
        ) from error

    if not header:
        raise ValueError(f"CSV file has no header: {file_path}")

    if len(header) < 2:
        raise ValueError(
            f"CSV file contains fewer than two columns: {file_path}"
        )

    joined_header = " ".join(header).lower()

    if "<html" in joined_header or "<!doctype" in joined_header:
        raise ValueError(
            f"The downloaded CSV appears to contain an HTML error page: "
            f"{file_path}"
        )


def validate_file(file_path: Path, file_format: FileFormat) -> None:
    """Validate a downloaded file according to its expected format."""
    if not file_path.exists():
        raise FileNotFoundError(f"File was not created: {file_path}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"Downloaded file is empty: {file_path}")

    if file_format == "json":
        validate_json(file_path)
    elif file_format == "csv":
        validate_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def download_file(
    session: requests.Session,
    url: str,
    output_path: Path,
    file_format: FileFormat,
    force: bool = False,
) -> None:
    """Download one file safely and validate its contents."""
    if output_path.exists() and not force:
        print(f"[SKIP] File already exists: {output_path}")
        print(f"       SHA-256: {calculate_sha256(output_path)}")
        return

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    print(f"[DOWNLOAD] {output_path.name}")
    print(f"           {url}")

    try:
        with session.get(
            url,
            stream=True,
            timeout=(15, 180),
        ) as response:
            response.raise_for_status()

            with temporary_path.open("wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        validate_file(temporary_path, file_format)

        # Atomic replacement after a successful download and validation.
        temporary_path.replace(output_path)

    except (requests.RequestException, OSError, ValueError) as error:
        if temporary_path.exists():
            temporary_path.unlink()

        raise RuntimeError(
            f"Could not download {output_path.name}: {error}"
        ) from error

    size_mb = output_path.stat().st_size / (1024 * 1024)
    checksum = calculate_sha256(output_path)

    print(f"[OK] Saved: {output_path}")
    print(f"     Size: {size_mb:.3f} MB")
    print(f"     SHA-256: {checksum}")


# ---------------------------------------------------------------------
# Main download workflow
# ---------------------------------------------------------------------

def download_mavedb_data(force: bool = False) -> None:
    """Download metadata, scores, and mapped variants from MaveDB."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MaveDB KCNH2 data acquisition")
    print("=" * 72)
    print(f"Score set: {SCORE_SET_URN}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "byte2beat-cardiovus/0.1 "
                "(research and educational use)"
            ),
            "Accept": "*/*",
        }
    )

    for filename, (url, file_format) in ENDPOINTS.items():
        output_path = OUTPUT_DIR / filename

        download_file(
            session=session,
            url=url,
            output_path=output_path,
            file_format=file_format,
            force=force,
        )

        print()

    print("=" * 72)
    print("Download completed successfully")
    print("=" * 72)

    for filename in ENDPOINTS:
        path = OUTPUT_DIR / filename

        print(
            f"{filename:<28} "
            f"{path.stat().st_size:>12,} bytes"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the selected KCNH2 score set from MaveDB."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that already exist.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the MaveDB download workflow."""
    args = parse_arguments()

    try:
        download_mavedb_data(force=args.force)
    except RuntimeError as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())