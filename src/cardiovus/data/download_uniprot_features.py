"""Download and normalize curated KCNH2 sequence features from UniProtKB.

Primary source
--------------
UniProtKB accession Q12809 (KCNH2_HUMAN)

Outputs
-------
data/raw/uniprot/Q12809.json
data/raw/uniprot/Q12809.gff
data/raw/uniprot/Q12809_download_metadata.json
data/external/kcnh2_uniprot_features.csv
data/external/kcnh2_domains.csv

The script also verifies that the canonical UniProt sequence is identical
to the NCBI RefSeq protein NP_000229.1 used by the project.

Usage
-----
From the repository root:

    python src/cardiovus/data/download_uniprot_features.py

Overwrite previously downloaded files:

    python src/cardiovus/data/download_uniprot_features.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from Bio import SeqIO


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

UNIPROT_ACCESSION = "Q12809"
EXPECTED_GENE = "KCNH2"
EXPECTED_ORGANISM = "Homo sapiens"
EXPECTED_LENGTH = 1159
NCBI_ACCESSION = "NP_000229.1"

UNIPROT_BASE_URL = "https://rest.uniprot.org/uniprotkb"
UNIPROT_JSON_URL = (
    f"{UNIPROT_BASE_URL}/{UNIPROT_ACCESSION}.json"
)
UNIPROT_GFF_URL = (
    f"{UNIPROT_BASE_URL}/{UNIPROT_ACCESSION}.gff"
)

ROOT_DIR = Path(__file__).resolve().parents[3]

RAW_UNIPROT_DIR = ROOT_DIR / "data" / "raw" / "uniprot"
EXTERNAL_DATA_DIR = ROOT_DIR / "data" / "external"
REFERENCE_FASTA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "reference"
    / f"{NCBI_ACCESSION}.fasta"
)

UNIPROT_JSON_PATH = (
    RAW_UNIPROT_DIR / f"{UNIPROT_ACCESSION}.json"
)
UNIPROT_GFF_PATH = (
    RAW_UNIPROT_DIR / f"{UNIPROT_ACCESSION}.gff"
)
DOWNLOAD_METADATA_PATH = (
    RAW_UNIPROT_DIR
    / f"{UNIPROT_ACCESSION}_download_metadata.json"
)

FEATURES_CSV_PATH = (
    EXTERNAL_DATA_DIR / "kcnh2_uniprot_features.csv"
)
DOMAINS_CSV_PATH = (
    EXTERNAL_DATA_DIR / "kcnh2_domains.csv"
)

# Sequence features relevant to domain-level biological analysis.
# Site-level annotations are intentionally excluded because the current
# objective is to compare larger regions rather than individual residues.
FEATURE_GROUP_MAP = {
    "Domain": "domain",
    "Region": "region",
    "Transmembrane": "membrane",
    "Intramembrane": "membrane",
    "Topological domain": "topology",
    "Motif": "motif",
    "Repeat": "repeat",
    "Coiled coil": "structure",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_bytes_atomically(
    content: bytes,
    output_path: Path,
) -> None:
    """Write bytes through a temporary file before replacing the target."""
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    temporary_path.write_bytes(content)
    temporary_path.replace(output_path)


def write_json_atomically(
    data: Any,
    output_path: Path,
) -> None:
    """Write JSON safely through a temporary file."""
    text = json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    temporary_path.write_text(
        text + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def download_file(
    session: requests.Session,
    url: str,
    output_path: Path,
    expected_content_types: tuple[str, ...],
    force: bool,
) -> dict[str, str | None]:
    """Download one UniProt resource and retain release headers."""
    if output_path.exists() and not force:
        print(f"[SKIP] File already exists: {output_path}")
        return {}

    print(f"[DOWNLOAD] {url}")

    response = session.get(
        url,
        timeout=(15, 180),
    )
    response.raise_for_status()

    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()

    if not any(
        accepted in content_type
        for accepted in expected_content_types
    ):
        raise ValueError(
            f"Unexpected content type for {url}: "
            f"{content_type!r}"
        )

    if not response.content:
        raise ValueError(
            f"UniProt returned an empty response: {url}"
        )

    write_bytes_atomically(
        response.content,
        output_path,
    )

    print(f"[OK] Saved: {output_path}")
    print(f"     SHA-256: {calculate_sha256(output_path)}")

    return {
        "x_uniprot_release": response.headers.get(
            "x-uniprot-release"
        ),
        "x_uniprot_release_date": response.headers.get(
            "x-uniprot-release-date"
        ),
        "etag": response.headers.get("etag"),
        "last_modified": response.headers.get(
            "last-modified"
        ),
    }


def extract_primary_gene(
    entry: dict[str, Any],
) -> str | None:
    """Extract the primary gene symbol from a UniProt JSON entry."""
    for gene_record in entry.get("genes", []):
        gene_name = gene_record.get("geneName")

        if isinstance(gene_name, dict):
            value = gene_name.get("value")

            if value:
                return str(value)

    return None


def extract_coordinate(
    location: dict[str, Any],
    boundary: str,
) -> tuple[int | None, str | None]:
    """Extract a UniProt feature coordinate and its modifier."""
    coordinate = location.get(boundary, {})

    if not isinstance(coordinate, dict):
        return None, None

    value = coordinate.get("value")
    modifier = coordinate.get("modifier")

    if value is None:
        return None, modifier

    return int(value), modifier


def extract_evidence_codes(
    feature: dict[str, Any],
) -> str:
    """Join evidence codes attached to one UniProt feature."""
    evidence_codes = sorted(
        {
            str(evidence.get("evidenceCode"))
            for evidence in feature.get("evidences", [])
            if evidence.get("evidenceCode")
        }
    )

    return ";".join(evidence_codes)


# ---------------------------------------------------------------------
# Biological validation
# ---------------------------------------------------------------------

def validate_uniprot_entry(
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Validate accession, gene, organism, length, and NCBI sequence."""
    primary_accession = entry.get(
        "primaryAccession"
    )
    gene = extract_primary_gene(entry)

    organism_record = entry.get("organism", {})
    organism = organism_record.get(
        "scientificName"
    )

    sequence_record = entry.get("sequence", {})
    uniprot_sequence = str(
        sequence_record.get("value", "")
    ).upper()
    declared_length = sequence_record.get(
        "length"
    )

    if primary_accession != UNIPROT_ACCESSION:
        raise ValueError(
            f"Unexpected UniProt accession: "
            f"{primary_accession}"
        )

    if gene != EXPECTED_GENE:
        raise ValueError(
            f"Unexpected gene: {gene}. "
            f"Expected {EXPECTED_GENE}."
        )

    if organism != EXPECTED_ORGANISM:
        raise ValueError(
            f"Unexpected organism: {organism}. "
            f"Expected {EXPECTED_ORGANISM}."
        )

    if (
        declared_length != EXPECTED_LENGTH
        or len(uniprot_sequence) != EXPECTED_LENGTH
    ):
        raise ValueError(
            "Unexpected UniProt sequence length: "
            f"declared={declared_length}, "
            f"observed={len(uniprot_sequence)}, "
            f"expected={EXPECTED_LENGTH}."
        )

    if not REFERENCE_FASTA_PATH.exists():
        raise FileNotFoundError(
            f"Missing NCBI reference FASTA: "
            f"{REFERENCE_FASTA_PATH}"
        )

    ncbi_record = SeqIO.read(
        REFERENCE_FASTA_PATH,
        "fasta",
    )
    ncbi_sequence = str(
        ncbi_record.seq
    ).upper()

    if ncbi_record.id != NCBI_ACCESSION:
        raise ValueError(
            f"Unexpected NCBI FASTA ID: "
            f"{ncbi_record.id}"
        )

    if uniprot_sequence != ncbi_sequence:
        mismatch_positions = [
            index + 1
            for index, (uniprot_aa, ncbi_aa)
            in enumerate(
                zip(
                    uniprot_sequence,
                    ncbi_sequence,
                )
            )
            if uniprot_aa != ncbi_aa
        ]

        raise ValueError(
            "UniProt Q12809 and NCBI NP_000229.1 "
            "are not identical. First mismatch "
            f"positions: {mismatch_positions[:10]}"
        )

    return {
        "uniprot_accession": primary_accession,
        "gene": gene,
        "organism": organism,
        "length": len(uniprot_sequence),
        "ncbi_accession": ncbi_record.id,
        "sequence_matches_ncbi": True,
    }


# ---------------------------------------------------------------------
# Feature parsing
# ---------------------------------------------------------------------

def parse_uniprot_features(
    entry: dict[str, Any],
) -> pd.DataFrame:
    """Convert relevant UniProt features into a tidy table."""
    rows: list[dict[str, Any]] = []

    for feature in entry.get("features", []):
        feature_type = feature.get("type")

        if feature_type not in FEATURE_GROUP_MAP:
            continue

        location = feature.get("location", {})
        start, start_modifier = (
            extract_coordinate(
                location,
                "start",
            )
        )
        end, end_modifier = extract_coordinate(
            location,
            "end",
        )

        if start is None or end is None:
            continue

        if start < 1 or end > EXPECTED_LENGTH:
            raise ValueError(
                f"Feature outside protein boundaries: "
                f"{feature_type} {start}-{end}"
            )

        if start > end:
            raise ValueError(
                f"Feature start is greater than end: "
                f"{feature_type} {start}-{end}"
            )

        description = feature.get(
            "description"
        )

        rows.append(
            {
                "uniprot_accession": (
                    UNIPROT_ACCESSION
                ),
                "gene": EXPECTED_GENE,
                "feature_type": feature_type,
                "feature_group": (
                    FEATURE_GROUP_MAP[
                        feature_type
                    ]
                ),
                "description": (
                    str(description).strip()
                    if description
                    else ""
                ),
                "start": start,
                "end": end,
                "length": end - start + 1,
                "start_modifier": start_modifier,
                "end_modifier": end_modifier,
                "feature_id": feature.get(
                    "featureId"
                ),
                "evidence_codes": (
                    extract_evidence_codes(
                        feature
                    )
                ),
                "source": "UniProtKB",
            }
        )

    feature_df = pd.DataFrame(rows)

    if feature_df.empty:
        raise ValueError(
            "No relevant UniProt features were found."
        )

    feature_df = feature_df.sort_values(
        [
            "feature_group",
            "start",
            "end",
            "feature_type",
        ]
    ).reset_index(drop=True)

    feature_df["base_label"] = np_where_description(
        feature_df
    )

    group_columns = [
        "feature_type",
        "base_label",
    ]

    feature_df["label_index"] = (
        feature_df.groupby(
            group_columns,
            dropna=False,
        ).cumcount()
        + 1
    )

    feature_df["label_count"] = (
        feature_df.groupby(
            group_columns,
            dropna=False,
        )["base_label"]
        .transform("count")
    )

    feature_df["feature_label"] = (
        feature_df.apply(
            build_feature_label,
            axis=1,
        )
    )

    return feature_df.drop(
        columns=[
            "base_label",
            "label_index",
            "label_count",
        ]
    )


def np_where_description(
    feature_df: pd.DataFrame,
) -> pd.Series:
    """Use the description when available, otherwise the feature type."""
    has_description = (
        feature_df["description"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    return feature_df[
        "description"
    ].where(
        has_description,
        feature_df["feature_type"],
    )


def build_feature_label(
    row: pd.Series,
) -> str:
    """Create a unique readable label for repeated features."""
    base_label = str(row["base_label"])

    if int(row["label_count"]) == 1:
        return base_label

    return (
        f"{base_label} "
        f"{int(row['label_index'])}"
    )


def save_feature_tables(
    feature_df: pd.DataFrame,
) -> None:
    """Save the complete and analysis-compatible feature tables."""
    feature_df.to_csv(
        FEATURES_CSV_PATH,
        index=False,
    )

    domain_groups = {
        "domain",
        "region",
        "membrane",
        "topology",
    }

    domains_df = feature_df.loc[
        feature_df[
            "feature_group"
        ].isin(domain_groups)
    ].copy()

    domains_df = domains_df.rename(
        columns={
            "feature_label": "domain",
        }
    )

    domain_columns = [
        "domain",
        "start",
        "end",
        "length",
        "feature_type",
        "feature_group",
        "description",
        "uniprot_accession",
        "gene",
        "feature_id",
        "evidence_codes",
        "source",
    ]

    domains_df[
        domain_columns
    ].to_csv(
        DOMAINS_CSV_PATH,
        index=False,
    )


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def run(force: bool) -> None:
    """Download, validate, parse, and save KCNH2 annotations."""
    RAW_UNIPROT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    EXTERNAL_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    release_headers: dict[
        str,
        str | None,
    ] = {}

    json_headers = download_file(
        session=session,
        url=UNIPROT_JSON_URL,
        output_path=UNIPROT_JSON_PATH,
        expected_content_types=(
            "application/json",
        ),
        force=force,
    )
    release_headers.update(json_headers)

    download_file(
        session=session,
        url=UNIPROT_GFF_URL,
        output_path=UNIPROT_GFF_PATH,
        expected_content_types=(
            "text/plain",
            "text/gff",
            "text/gff3",
        ),
        force=force,
    )

    with UNIPROT_JSON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        entry = json.load(file)

    validation = validate_uniprot_entry(
        entry
    )
    feature_df = parse_uniprot_features(
        entry
    )
    save_feature_tables(feature_df)

    processing_metadata = {
        **validation,
        "uniprot_json_url": (
            UNIPROT_JSON_URL
        ),
        "uniprot_gff_url": (
            UNIPROT_GFF_URL
        ),
        "processed_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "release_headers": (
            release_headers
        ),
        "files": {
            "json": {
                "path": str(
                    UNIPROT_JSON_PATH.relative_to(
                        ROOT_DIR
                    )
                ),
                "sha256": calculate_sha256(
                    UNIPROT_JSON_PATH
                ),
            },
            "gff": {
                "path": str(
                    UNIPROT_GFF_PATH.relative_to(
                        ROOT_DIR
                    )
                ),
                "sha256": calculate_sha256(
                    UNIPROT_GFF_PATH
                ),
            },
            "features_csv": {
                "path": str(
                    FEATURES_CSV_PATH.relative_to(
                        ROOT_DIR
                    )
                ),
                "sha256": calculate_sha256(
                    FEATURES_CSV_PATH
                ),
            },
            "domains_csv": {
                "path": str(
                    DOMAINS_CSV_PATH.relative_to(
                        ROOT_DIR
                    )
                ),
                "sha256": calculate_sha256(
                    DOMAINS_CSV_PATH
                ),
            },
        },
        "feature_counts": {
            str(label): int(count)
            for label, count
            in feature_df[
                "feature_group"
            ].value_counts().items()
        },
    }

    write_json_atomically(
        processing_metadata,
        DOWNLOAD_METADATA_PATH,
    )

    print()
    print("=" * 72)
    print("UniProt KCNH2 feature acquisition completed")
    print("=" * 72)
    print(
        f"UniProt: {validation['uniprot_accession']}"
    )
    print(f"Gene: {validation['gene']}")
    print(
        f"Length: {validation['length']} aa"
    )
    print(
        "Matches NP_000229.1: "
        f"{validation['sequence_matches_ncbi']}"
    )
    print(
        f"Features retained: {len(feature_df)}"
    )
    print()
    print(
        feature_df[
            "feature_group"
        ].value_counts()
    )
    print()
    print(f"Features CSV: {FEATURES_CSV_PATH}")
    print(f"Domains CSV:  {DOMAINS_CSV_PATH}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download and normalize curated "
            "KCNH2 features from UniProtKB."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite downloaded UniProt "
            "files."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line workflow."""
    arguments = parse_arguments()

    try:
        run(force=arguments.force)
    except (
        requests.RequestException,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"\n[ERROR] {error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
