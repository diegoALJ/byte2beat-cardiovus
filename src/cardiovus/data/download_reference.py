"""Download and validate the KCNH2 protein reference from NCBI.

The script downloads:

- NP_000229.1.fasta
- NP_000229.1.gp
- NP_000229.1_metadata.json

The GenPept record is downloaded because the FASTA header alone is not
sufficient for robust verification of the annotated gene symbol.

Usage
-----
From the repository root:

    python src/cardiovus/data/download_reference.py \
        --email your-email@example.com

Overwrite existing files:

    python src/cardiovus/data/download_reference.py \
        --email your-email@example.com \
        --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from Bio import SeqIO


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ACCESSION = "NP_000229.1"
EXPECTED_GENE = "KCNH2"
EXPECTED_LENGTH = 1159
EXPECTED_ORGANISM = "Homo sapiens"

EFETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)

# File location:
# repository/src/cardiovus/data/download_reference.py
# parents[3] points to the repository root.
ROOT_DIR = Path(__file__).resolve().parents[3]

OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "reference"

FASTA_PATH = OUTPUT_DIR / f"{ACCESSION}.fasta"
GENPEPT_PATH = OUTPUT_DIR / f"{ACCESSION}.gp"
METADATA_PATH = OUTPUT_DIR / f"{ACCESSION}_metadata.json"

TOOL_NAME = "byte2beat_cardiovus"


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def write_text_atomically(text: str, output_path: Path) -> None:
    """Write text to a temporary file and replace the target atomically."""
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(output_path)


def request_ncbi_record(
    session: requests.Session,
    rettype: str,
    email: str | None,
    api_key: str | None,
) -> str:
    """Retrieve a protein record from NCBI EFetch."""
    parameters = {
        "db": "protein",
        "id": ACCESSION,
        "rettype": rettype,
        "retmode": "text",
        "tool": TOOL_NAME,
    }

    if email:
        parameters["email"] = email

    if api_key:
        parameters["api_key"] = api_key

    response = session.get(
        EFETCH_URL,
        params=parameters,
        timeout=(15, 180),
    )

    response.raise_for_status()

    text = response.text.strip()

    if not text:
        raise ValueError(
            f"NCBI returned an empty response for rettype={rettype}."
        )

    possible_error_tokens = (
        "<ERROR>",
        "Error occurred",
        "Failed to retrieve",
        "Cannot process",
    )

    if any(token in text for token in possible_error_tokens):
        raise ValueError(
            f"NCBI returned an error for rettype={rettype}:\n{text[:500]}"
        )

    return text + "\n"


# ---------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------

def download_reference_files(
    session: requests.Session,
    email: str | None,
    api_key: str | None,
    force: bool,
) -> None:
    """Download FASTA and GenPept records."""
    resources = (
        ("FASTA", "fasta", FASTA_PATH),
        ("GenPept", "gp", GENPEPT_PATH),
    )

    for label, rettype, output_path in resources:
        if output_path.exists() and not force:
            print(f"[SKIP] {label} already exists: {output_path}")
            continue

        print(f"[DOWNLOAD] {label}: {ACCESSION}")

        record_text = request_ncbi_record(
            session=session,
            rettype=rettype,
            email=email,
            api_key=api_key,
        )

        write_text_atomically(record_text, output_path)

        print(f"[OK] Saved: {output_path}")
        print(f"     SHA-256: {calculate_sha256(output_path)}")


# ---------------------------------------------------------------------
# Biological validation
# ---------------------------------------------------------------------

def extract_gene_symbols(genpept_record: Any) -> list[str]:
    """Extract gene symbols from the GenPept feature annotations."""
    gene_symbols: set[str] = set()

    for feature in genpept_record.features:
        genes = feature.qualifiers.get("gene", [])

        for gene in genes:
            normalized_gene = gene.strip()

            if normalized_gene:
                gene_symbols.add(normalized_gene)

    return sorted(gene_symbols)


def load_and_validate_reference() -> dict[str, Any]:
    """Load the downloaded records and validate ID, length, and gene."""
    if not FASTA_PATH.exists():
        raise FileNotFoundError(f"Missing FASTA file: {FASTA_PATH}")

    if not GENPEPT_PATH.exists():
        raise FileNotFoundError(
            f"Missing GenPept file: {GENPEPT_PATH}"
        )

    fasta_record = SeqIO.read(FASTA_PATH, "fasta")
    genpept_record = SeqIO.read(GENPEPT_PATH, "genbank")

    fasta_id = fasta_record.id
    genpept_id = genpept_record.id

    fasta_length = len(fasta_record.seq)
    genpept_length = len(genpept_record.seq)

    gene_symbols = extract_gene_symbols(genpept_record)

    organism = genpept_record.annotations.get(
        "organism",
        "Unknown",
    )

    # -------------------------------------------------------------
    # Accession validation
    # -------------------------------------------------------------
    if fasta_id != ACCESSION:
        raise ValueError(
            f"Unexpected FASTA ID: {fasta_id}. "
            f"Expected: {ACCESSION}."
        )

    if genpept_id != ACCESSION:
        raise ValueError(
            f"Unexpected GenPept ID: {genpept_id}. "
            f"Expected: {ACCESSION}."
        )

    # -------------------------------------------------------------
    # Length validation
    # -------------------------------------------------------------
    if fasta_length != EXPECTED_LENGTH:
        raise ValueError(
            f"Unexpected FASTA length: {fasta_length}. "
            f"Expected: {EXPECTED_LENGTH}."
        )

    if genpept_length != EXPECTED_LENGTH:
        raise ValueError(
            f"Unexpected GenPept length: {genpept_length}. "
            f"Expected: {EXPECTED_LENGTH}."
        )

    # -------------------------------------------------------------
    # Sequence consistency
    # -------------------------------------------------------------
    if str(fasta_record.seq) != str(genpept_record.seq):
        raise ValueError(
            "The FASTA and GenPept protein sequences are different."
        )

    # -------------------------------------------------------------
    # Gene validation
    # -------------------------------------------------------------
    if EXPECTED_GENE not in gene_symbols:
        raise ValueError(
            f"Expected gene {EXPECTED_GENE} was not found. "
            f"Gene annotations found: {gene_symbols}"
        )

    # -------------------------------------------------------------
    # Organism validation
    # -------------------------------------------------------------
    if organism != EXPECTED_ORGANISM:
        raise ValueError(
            f"Unexpected organism: {organism}. "
            f"Expected: {EXPECTED_ORGANISM}."
        )

    return {
        "accession": fasta_id,
        "gene": EXPECTED_GENE,
        "gene_annotations": gene_symbols,
        "length": fasta_length,
        "organism": organism,
        "protein_description": fasta_record.description,
        "sequence_matches_genpept": True,
        "expected_accession": ACCESSION,
        "expected_gene": EXPECTED_GENE,
        "expected_length": EXPECTED_LENGTH,
        "expected_organism": EXPECTED_ORGANISM,
        "fasta_filename": FASTA_PATH.name,
        "genpept_filename": GENPEPT_PATH.name,
        "fasta_sha256": calculate_sha256(FASTA_PATH),
        "genpept_sha256": calculate_sha256(GENPEPT_PATH),
        "metadata_generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": "NCBI Protein / RefSeq",
        "retrieval_method": "NCBI EFetch",
    }


def save_metadata(metadata: dict[str, Any]) -> None:
    """Save validated reference metadata as JSON."""
    metadata_text = json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
    )

    write_text_atomically(
        metadata_text + "\n",
        METADATA_PATH,
    )


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def download_and_validate_reference(
    email: str | None,
    api_key: str | None,
    force: bool,
) -> None:
    """Download and validate the KCNH2 reference."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("NCBI KCNH2 reference acquisition")
    print("=" * 72)
    print(f"Accession: {ACCESSION}")
    print(f"Expected gene: {EXPECTED_GENE}")
    print(f"Expected length: {EXPECTED_LENGTH} aa")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    if not email:
        print(
            "[WARNING] No NCBI contact email was provided. "
            "Use --email or set NCBI_EMAIL."
        )
        print()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "byte2beat-cardiovus/0.1 "
                f"(contact: {email or 'not-provided'})"
            )
        }
    )

    download_reference_files(
        session=session,
        email=email,
        api_key=api_key,
        force=force,
    )

    print()
    print("[VALIDATE] Checking accession, length, gene, and sequence")

    metadata = load_and_validate_reference()
    save_metadata(metadata)

    print()
    print("=" * 72)
    print("Reference validated successfully")
    print("=" * 72)
    print(f"ID:       {metadata['accession']}")
    print(f"Length:   {metadata['length']} aa")
    print(f"Gene:     {metadata['gene']}")
    print(f"Organism: {metadata['organism']}")
    print()
    print(f"FASTA:    {FASTA_PATH}")
    print(f"GenPept:  {GENPEPT_PATH}")
    print(f"Metadata: {METADATA_PATH}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download and validate the KCNH2 RefSeq protein "
            "reference from NCBI."
        )
    )

    parser.add_argument(
        "--email",
        default=os.getenv("NCBI_EMAIL"),
        help=(
            "Contact email sent to NCBI. It can also be supplied "
            "through the NCBI_EMAIL environment variable."
        ),
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv("NCBI_API_KEY"),
        help=(
            "Optional NCBI API key. It can also be supplied through "
            "the NCBI_API_KEY environment variable."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing reference files.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the reference download workflow."""
    arguments = parse_arguments()

    try:
        download_and_validate_reference(
            email=arguments.email,
            api_key=arguments.api_key,
            force=arguments.force,
        )
    except (
        requests.RequestException,
        OSError,
        ValueError,
    ) as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())