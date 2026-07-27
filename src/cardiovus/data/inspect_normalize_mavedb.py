"""Inspect and normalize the KCNH2 MaveDB score set.

The script performs:

1. Initial inspection of the raw score CSV.
2. MAVE-HGVS protein variant classification.
3. Parsing of simple missense substitutions.
4. Validation against the NP_000229.1 FASTA sequence.
5. Preliminary score-distribution summary.
6. Creation of QC and normalized Parquet datasets.

Outputs
-------
data/interim/kcnh2_variants_qc.parquet
data/interim/kcnh2_variants_normalized.parquet

reports/data_quality/initial_inspection.json
reports/data_quality/column_dtypes.csv
reports/data_quality/missing_values.csv

Optional outputs, created only when applicable:
reports/data_quality/protein_variant_duplicates.csv
reports/data_quality/reference_mismatches.csv

Usage
-----
Run from the repository root:

    python src/cardiovus/data/inspect_normalize_mavedb.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from Bio import SeqIO


# ---------------------------------------------------------------------
# Paths and biological reference
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[3]

MAVEDB_DIR = ROOT_DIR / "data" / "raw" / "mavedb"
REFERENCE_DIR = ROOT_DIR / "data" / "raw" / "reference"

SCORES_PATH = MAVEDB_DIR / "scores.csv"
METADATA_PATH = MAVEDB_DIR / "score_set_metadata.json"
MAPPED_VARIANTS_PATH = MAVEDB_DIR / "mapped_variants.json"

FASTA_PATH = REFERENCE_DIR / "NP_000229.1.fasta"

INTERIM_DIR = ROOT_DIR / "data" / "interim"
REPORT_DIR = ROOT_DIR / "reports" / "data_quality"

QC_DATASET_PATH = (
    INTERIM_DIR / "kcnh2_variants_qc.parquet"
)

NORMALIZED_DATASET_PATH = (
    INTERIM_DIR / "kcnh2_variants_normalized.parquet"
)

INSPECTION_REPORT_PATH = (
    REPORT_DIR / "initial_inspection.json"
)

DTYPES_REPORT_PATH = REPORT_DIR / "column_dtypes.csv"
MISSING_REPORT_PATH = REPORT_DIR / "missing_values.csv"

DUPLICATES_REPORT_PATH = (
    REPORT_DIR / "protein_variant_duplicates.csv"
)

MISMATCHES_REPORT_PATH = (
    REPORT_DIR / "reference_mismatches.csv"
)

EXPECTED_ACCESSION = "NP_000229.1"
EXPECTED_GENE = "KCNH2"
EXPECTED_LENGTH = 1159


# ---------------------------------------------------------------------
# Amino-acid mappings
# ---------------------------------------------------------------------

AA3_TO_AA1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Sec": "U",
    "Pyl": "O",
    "Ter": "*",
}


# ---------------------------------------------------------------------
# MAVE-HGVS regular expressions
# ---------------------------------------------------------------------

SIMPLE_SUBSTITUTION_PATTERN = re.compile(
    r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter)$"
)

POSITIONAL_EQUALITY_PATTERN = re.compile(
    r"^p\.([A-Z][a-z]{2})(\d+)=$"
)

MULTI_VARIANT_PATTERN = re.compile(
    r"^p\.\[(.+)\]$"
)


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def require_files(paths: list[Path]) -> None:
    """Raise an error when one or more required files are missing."""
    missing = [path for path in paths if not path.exists()]

    if missing:
        missing_text = "\n".join(
            f"  - {path}" for path in missing
        )

        raise FileNotFoundError(
            f"Required files are missing:\n{missing_text}"
        )


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def json_default(value: Any) -> Any:
    """Convert pandas and NumPy scalar values to JSON-compatible types."""
    if hasattr(value, "item"):
        return value.item()

    if pd.isna(value):
        return None

    return str(value)


def summarize_json_structure(data: Any) -> dict[str, Any]:
    """Generate a lightweight structural summary of a JSON object."""
    if isinstance(data, dict):
        return {
            "top_level_type": "dictionary",
            "top_level_key_count": len(data),
            "top_level_keys": list(data.keys())[:100],
        }

    if isinstance(data, list):
        return {
            "top_level_type": "list",
            "top_level_length": len(data),
            "first_item_type": (
                type(data[0]).__name__ if data else None
            ),
        }

    return {
        "top_level_type": type(data).__name__,
    }


# ---------------------------------------------------------------------
# Variant normalization and classification
# ---------------------------------------------------------------------

def normalize_protein_hgvs(value: Any) -> str | None:
    """Normalize whitespace and remove an optional target prefix.

    Examples
    --------
    p.Ala561Val            -> p.Ala561Val
    KCNH2:p.Ala561Val      -> p.Ala561Val
    NP_000229.1:p.Ala561Val -> p.Ala561Val
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    protein_start = text.rfind("p.")

    if protein_start >= 0:
        return text[protein_start:]

    return text


def classify_protein_variant(hgvs_pro: str | None) -> str:
    """Classify a normalized protein MAVE-HGVS variant."""
    if hgvs_pro is None:
        return "missing"

    if hgvs_pro in {"p.=", "p.(=)"}:
        return "wild_type_or_synonymous_summary"

    if MULTI_VARIANT_PATTERN.fullmatch(hgvs_pro):
        return "multi_variant"

    positional_equality = POSITIONAL_EQUALITY_PATTERN.fullmatch(
        hgvs_pro
    )

    if positional_equality:
        return "synonymous"

    substitution = SIMPLE_SUBSTITUTION_PATTERN.fullmatch(
        hgvs_pro
    )

    if substitution:
        wt_aa3, _, mut_aa3 = substitution.groups()

        if mut_aa3 == "Ter":
            return "stop_gained"

        if wt_aa3 == mut_aa3:
            return "synonymous"

        return "simple_missense"

    if "fs" in hgvs_pro:
        return "frameshift"

    if "delins" in hgvs_pro:
        return "deletion_insertion"

    if "del" in hgvs_pro:
        return "deletion"

    if "dup" in hgvs_pro:
        return "duplication"

    if "ins" in hgvs_pro:
        return "insertion"

    return "other_or_unparsed"


def count_variant_components(
    hgvs_pro: str | None,
) -> int | None:
    """Count the number of changes represented by a variant string."""
    if hgvs_pro is None:
        return None

    multi_match = MULTI_VARIANT_PATTERN.fullmatch(hgvs_pro)

    if multi_match:
        content = multi_match.group(1)
        return content.count(";") + 1

    return 1


def parse_simple_missense(
    hgvs_pro: str | None,
) -> dict[str, Any]:
    """Parse a simple missense substitution.

    Example
    -------
    p.Ala561Val becomes:

    wt_aa3=Ala
    wt_aa1=A
    position=561
    mut_aa3=Val
    mut_aa1=V
    variant_id=A561V
    """
    empty_result = {
        "wt_aa3": None,
        "wt_aa1": None,
        "position": None,
        "mut_aa3": None,
        "mut_aa1": None,
        "variant_id": None,
    }

    if hgvs_pro is None:
        return empty_result

    match = SIMPLE_SUBSTITUTION_PATTERN.fullmatch(hgvs_pro)

    if not match:
        return empty_result

    wt_aa3, position_text, mut_aa3 = match.groups()

    if mut_aa3 == "Ter" or wt_aa3 == mut_aa3:
        return empty_result

    wt_aa1 = AA3_TO_AA1.get(wt_aa3)
    mut_aa1 = AA3_TO_AA1.get(mut_aa3)

    if wt_aa1 is None or mut_aa1 is None:
        return empty_result

    position = int(position_text)

    return {
        "wt_aa3": wt_aa3,
        "wt_aa1": wt_aa1,
        "position": position,
        "mut_aa3": mut_aa3,
        "mut_aa1": mut_aa1,
        "variant_id": f"{wt_aa1}{position}{mut_aa1}",
    }


# ---------------------------------------------------------------------
# Reference validation
# ---------------------------------------------------------------------

def load_reference_sequence() -> tuple[str, str]:
    """Load and validate the NCBI reference FASTA."""
    record = SeqIO.read(FASTA_PATH, "fasta")

    if record.id != EXPECTED_ACCESSION:
        raise ValueError(
            f"Unexpected FASTA ID: {record.id}. "
            f"Expected {EXPECTED_ACCESSION}."
        )

    sequence = str(record.seq).upper()

    if len(sequence) != EXPECTED_LENGTH:
        raise ValueError(
            f"Unexpected FASTA length: {len(sequence)}. "
            f"Expected {EXPECTED_LENGTH}."
        )

    return record.id, sequence


def validate_variant_against_reference(
    row: pd.Series,
    reference_sequence: str,
) -> pd.Series:
    """Check whether a simple missense variant matches the FASTA."""
    position = row["position"]
    expected_wt = row["wt_aa1"]

    if pd.isna(position) or expected_wt is None:
        return pd.Series(
            {
                "reference_position_valid": None,
                "reference_residue": None,
                "ref_match": None,
            }
        )

    position = int(position)

    if position < 1 or position > len(reference_sequence):
        return pd.Series(
            {
                "reference_position_valid": False,
                "reference_residue": None,
                "ref_match": False,
            }
        )

    reference_residue = reference_sequence[position - 1]

    return pd.Series(
        {
            "reference_position_valid": True,
            "reference_residue": reference_residue,
            "ref_match": reference_residue == expected_wt,
        }
    )


# ---------------------------------------------------------------------
# Score inspection
# ---------------------------------------------------------------------

def describe_score(score: pd.Series) -> dict[str, Any]:
    """Create a preliminary statistical summary of the score."""
    clean_score = score.dropna()

    if clean_score.empty:
        return {
            "non_missing_scores": 0,
            "message": "No numeric scores were available.",
        }

    description = clean_score.describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    ).to_dict()

    q1 = clean_score.quantile(0.25)
    q3 = clean_score.quantile(0.75)
    iqr = q3 - q1

    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)

    iqr_outliers = (
        (clean_score < lower_limit)
        | (clean_score > upper_limit)
    ).sum()

    return {
        "descriptive_statistics": description,
        "skewness": clean_score.skew(),
        "kurtosis": clean_score.kurtosis(),
        "iqr": iqr,
        "iqr_lower_limit": lower_limit,
        "iqr_upper_limit": upper_limit,
        "iqr_outlier_count": int(iqr_outliers),
        "unique_numeric_scores": int(clean_score.nunique()),
        "score_direction_interpreted": False,
        "note": (
            "These statistics describe scale and distribution only. "
            "The biological direction of the score must be determined "
            "from the experiment metadata and publication."
        ),
    }


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def inspect_and_normalize() -> None:
    """Run the complete initial inspection and normalization."""
    require_files(
        [
            SCORES_PATH,
            METADATA_PATH,
            MAPPED_VARIANTS_PATH,
            FASTA_PATH,
        ]
    )

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("KCNH2 MaveDB initial inspection and normalization")
    print("=" * 72)

    # -------------------------------------------------------------
    # Load raw files
    # -------------------------------------------------------------
    scores = pd.read_csv(
        SCORES_PATH,
        na_values=["NA", "NaN", "nan", "None", "null"],
        keep_default_na=True,
        low_memory=False,
    )

    metadata = load_json(METADATA_PATH)
    mapped_variants = load_json(MAPPED_VARIANTS_PATH)

    fasta_id, reference_sequence = load_reference_sequence()

    print(f"Scores dimensions: {scores.shape}")
    print(f"Reference: {fasta_id}")
    print(f"Reference length: {len(reference_sequence)} aa")

    # -------------------------------------------------------------
    # Validate required MaveDB columns
    # -------------------------------------------------------------
    required_columns = {"score", "hgvs_pro"}
    missing_required = required_columns - set(scores.columns)

    if missing_required:
        raise ValueError(
            "The score CSV does not contain the required columns "
            f"for this pipeline: {sorted(missing_required)}. "
            f"Columns found: {list(scores.columns)}"
        )

    # -------------------------------------------------------------
    # Preserve and normalize the experimental score
    # -------------------------------------------------------------
    scores["score_numeric"] = pd.to_numeric(
        scores["score"],
        errors="coerce",
    )

    nonempty_original_scores = scores["score"].notna()

    score_parse_failure = (
        nonempty_original_scores
        & scores["score_numeric"].isna()
    )

    # -------------------------------------------------------------
    # Normalize and classify protein variants
    # -------------------------------------------------------------
    scores["hgvs_pro_normalized"] = scores[
        "hgvs_pro"
    ].apply(normalize_protein_hgvs)

    scores["variant_class"] = scores[
        "hgvs_pro_normalized"
    ].apply(classify_protein_variant)

    scores["variant_component_count"] = scores[
        "hgvs_pro_normalized"
    ].apply(count_variant_components)

    parsed_variants = scores[
        "hgvs_pro_normalized"
    ].apply(parse_simple_missense)

    parsed_dataframe = pd.DataFrame(
        parsed_variants.tolist(),
        index=scores.index,
    )

    scores = pd.concat(
        [scores, parsed_dataframe],
        axis=1,
    )

    # -------------------------------------------------------------
    # Validate variants against NP_000229.1
    # -------------------------------------------------------------
    reference_validation = scores.apply(
        validate_variant_against_reference,
        axis=1,
        reference_sequence=reference_sequence,
    )

    scores = pd.concat(
        [scores, reference_validation],
        axis=1,
    )

    # -------------------------------------------------------------
    # Detect duplicate protein consequences
    # -------------------------------------------------------------
    scores["protein_variant_duplicate"] = (
        scores["variant_id"].notna()
        & scores["variant_id"].duplicated(keep=False)
    )

    # -------------------------------------------------------------
    # Define eligibility for the first modeling dataset
    # -------------------------------------------------------------
    scores["eligible_simple_missense"] = (
        scores["variant_class"].eq("simple_missense")
        & scores["score_numeric"].notna()
        & scores["ref_match"].eq(True)
    )

    # -------------------------------------------------------------
    # Save full QC dataset
    # -------------------------------------------------------------
    scores.to_parquet(
        QC_DATASET_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Create normalized simple-missense dataset
    # -------------------------------------------------------------
    normalized = scores.loc[
        scores["eligible_simple_missense"]
    ].copy()

    generated_columns = [
        "variant_id",
        "hgvs_pro_normalized",
        "wt_aa3",
        "wt_aa1",
        "position",
        "mut_aa3",
        "mut_aa1",
        "score_numeric",
        "reference_residue",
        "ref_match",
        "protein_variant_duplicate",
    ]

    original_columns = [
        column
        for column in scores.columns
        if column not in generated_columns
    ]

    normalized = normalized[
        generated_columns + original_columns
    ]

    normalized.to_parquet(
        NORMALIZED_DATASET_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Save dtype and missing-value reports
    # -------------------------------------------------------------
    dtype_report = pd.DataFrame(
        {
            "column": scores.columns,
            "dtype": [
                str(dtype)
                for dtype in scores.dtypes
            ],
        }
    )

    dtype_report.to_csv(
        DTYPES_REPORT_PATH,
        index=False,
    )

    missing_report = pd.DataFrame(
        {
            "column": scores.columns,
            "missing_count": scores.isna().sum().values,
            "missing_percentage": (
                scores.isna().mean().values * 100
            ),
        }
    ).sort_values(
        "missing_percentage",
        ascending=False,
    )

    missing_report.to_csv(
        MISSING_REPORT_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Save duplicate and reference mismatch reports
    # -------------------------------------------------------------
    duplicates = scores.loc[
        scores["protein_variant_duplicate"]
    ].copy()

    if not duplicates.empty:
        duplicate_columns = [
            column
            for column in [
                "hgvs_nt",
                "hgvs_pro",
                "hgvs_pro_normalized",
                "variant_id",
                "score",
                "score_numeric",
            ]
            if column in duplicates.columns
        ]

        duplicates[
            duplicate_columns
        ].sort_values(
            "variant_id"
        ).to_csv(
            DUPLICATES_REPORT_PATH,
            index=False,
        )
    elif DUPLICATES_REPORT_PATH.exists():
        DUPLICATES_REPORT_PATH.unlink()

    reference_mismatches = scores.loc[
        scores["variant_class"].eq("simple_missense")
        & scores["ref_match"].eq(False)
    ].copy()

    if not reference_mismatches.empty:
        mismatch_columns = [
            column
            for column in [
                "hgvs_nt",
                "hgvs_pro",
                "variant_id",
                "position",
                "wt_aa1",
                "reference_residue",
                "score_numeric",
            ]
            if column in reference_mismatches.columns
        ]

        reference_mismatches[
            mismatch_columns
        ].to_csv(
            MISMATCHES_REPORT_PATH,
            index=False,
        )
    elif MISMATCHES_REPORT_PATH.exists():
        MISMATCHES_REPORT_PATH.unlink()

    # -------------------------------------------------------------
    # Build JSON inspection report
    # -------------------------------------------------------------
    variant_class_counts = {
        str(label): int(count)
        for label, count in scores[
            "variant_class"
        ].value_counts(
            dropna=False
        ).items()
    }

    metadata_text = json.dumps(
        metadata,
        ensure_ascii=False,
    )

    inspection_report = {
        "raw_files": {
            "scores_csv": str(
                SCORES_PATH.relative_to(ROOT_DIR)
            ),
            "metadata_json": str(
                METADATA_PATH.relative_to(ROOT_DIR)
            ),
            "mapped_variants_json": str(
                MAPPED_VARIANTS_PATH.relative_to(ROOT_DIR)
            ),
            "reference_fasta": str(
                FASTA_PATH.relative_to(ROOT_DIR)
            ),
        },
        "reference": {
            "accession": fasta_id,
            "expected_gene": EXPECTED_GENE,
            "length": len(reference_sequence),
        },
        "scores_csv": {
            "rows": int(scores.shape[0]),
            "columns": int(scores.shape[1]),
            "column_names": list(scores.columns),
            "original_column_names": list(
                pd.read_csv(
                    SCORES_PATH,
                    nrows=0,
                ).columns
            ),
            "numeric_score_count": int(
                scores["score_numeric"].notna().sum()
            ),
            "missing_score_count": int(
                scores["score_numeric"].isna().sum()
            ),
            "score_parse_failure_count": int(
                score_parse_failure.sum()
            ),
            "unique_hgvs_pro": int(
                scores[
                    "hgvs_pro_normalized"
                ].nunique(
                    dropna=True
                )
            ),
            "unique_simple_missense_ids": int(
                scores["variant_id"].nunique(
                    dropna=True
                )
            ),
            "duplicate_simple_missense_rows": int(
                scores[
                    "protein_variant_duplicate"
                ].sum()
            ),
            "variant_class_counts": variant_class_counts,
            "multi_variant_rows": int(
                scores[
                    "variant_class"
                ].eq("multi_variant").sum()
            ),
            "simple_missense_rows": int(
                scores[
                    "variant_class"
                ].eq("simple_missense").sum()
            ),
            "reference_match_count": int(
                scores["ref_match"].eq(True).sum()
            ),
            "reference_mismatch_count": int(
                scores["ref_match"].eq(False).sum()
            ),
            "eligible_normalized_rows": int(
                scores[
                    "eligible_simple_missense"
                ].sum()
            ),
            "score_summary": describe_score(
                scores["score_numeric"]
            ),
        },
        "score_set_metadata": {
            "json_structure": summarize_json_structure(
                metadata
            ),
            "contains_KCNH2_text": (
                EXPECTED_GENE in metadata_text
            ),
            "contains_reference_accession_text": (
                EXPECTED_ACCESSION in metadata_text
            ),
        },
        "mapped_variants": {
            "json_structure": summarize_json_structure(
                mapped_variants
            ),
            "note": (
                "Mapped VRS variants are inspected structurally here. "
                "They are not yet merged into the normalized table."
            ),
        },
        "outputs": {
            "qc_dataset": str(
                QC_DATASET_PATH.relative_to(ROOT_DIR)
            ),
            "normalized_dataset": str(
                NORMALIZED_DATASET_PATH.relative_to(ROOT_DIR)
            ),
        },
    }

    with INSPECTION_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            inspection_report,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------
    print()
    print("=" * 72)
    print("Initial inspection summary")
    print("=" * 72)
    print(f"Rows in raw CSV: {len(scores):,}")
    print(
        "Numeric scores: "
        f"{scores['score_numeric'].notna().sum():,}"
    )
    print(
        "Unique protein variants: "
        f"{scores['hgvs_pro_normalized'].nunique():,}"
    )
    print(
        "Simple missense rows: "
        f"{scores['variant_class'].eq('simple_missense').sum():,}"
    )
    print(
        "Multi-variant rows: "
        f"{scores['variant_class'].eq('multi_variant').sum():,}"
    )
    print(
        "Reference matches: "
        f"{scores['ref_match'].eq(True).sum():,}"
    )
    print(
        "Reference mismatches: "
        f"{scores['ref_match'].eq(False).sum():,}"
    )
    print(
        "Eligible normalized rows: "
        f"{len(normalized):,}"
    )

    print()
    print("Variant classes:")

    for label, count in variant_class_counts.items():
        print(f"  {label:<35} {count:>8,}")

    print()
    print("Outputs:")
    print(f"  QC dataset:         {QC_DATASET_PATH}")
    print(f"  Normalized dataset: {NORMALIZED_DATASET_PATH}")
    print(f"  Inspection report:  {INSPECTION_REPORT_PATH}")
    print(f"  Missing values:     {MISSING_REPORT_PATH}")
    print(f"  Column dtypes:      {DTYPES_REPORT_PATH}")


def main() -> int:
    """Run the inspection and normalization workflow."""
    try:
        inspect_and_normalize()
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        pd.errors.ParserError,
    ) as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())