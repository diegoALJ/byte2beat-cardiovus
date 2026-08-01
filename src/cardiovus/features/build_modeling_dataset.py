"""Build the model-ready KCNH2 feature dataset.

This script converts the validated missense dataset with UniProt domain
annotations into a compact, leakage-safe table for three models:

Model 0
    Constant median baseline. No predictive features are required.

Model 1
    XGBoost using biochemical, substitution, and positional features.

Model 2
    XGBoost using Model 1 features plus curated UniProt domain features.

ESM-2 features are intentionally excluded. They can later be merged using
``position`` as the stable key because the planned MVP uses one WT contextual
embedding per residue.

Input
-----
data/interim/kcnh2_variants_with_domains.parquet

Outputs
-------
data/processed/kcnh2_modeling_features.parquet
data/processed/kcnh2_modeling_features.csv.gz
data/processed/kcnh2_modeling_feature_schema.json
data/processed/kcnh2_modeling_dataset_summary.json

Usage
-----
Run from the repository root:

    python src/cardiovus/features/build_modeling_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------------------
# Central configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CFG:
    """Central configuration for reproducible feature generation."""

    ROOT_DIR: Path = Path(__file__).resolve().parents[3]

    INPUT_PATH: Path = (
        ROOT_DIR
        / "data"
        / "interim"
        / "kcnh2_variants_with_domains.parquet"
    )

    OUTPUT_DIR: Path = ROOT_DIR / "data" / "processed"

    OUTPUT_PARQUET_PATH: Path = (
        OUTPUT_DIR / "kcnh2_modeling_features.parquet"
    )
    OUTPUT_CSV_PATH: Path = (
        OUTPUT_DIR / "kcnh2_modeling_features.csv.gz"
    )
    FEATURE_SCHEMA_PATH: Path = (
        OUTPUT_DIR / "kcnh2_modeling_feature_schema.json"
    )
    DATASET_SUMMARY_PATH: Path = (
        OUTPUT_DIR / "kcnh2_modeling_dataset_summary.json"
    )

    TARGET_COLUMN: str = "score_numeric"
    GROUP_COLUMN: str = "position"
    EXPECTED_ROWS: int = 20_683
    EXPECTED_PROTEIN_LENGTH: int = 1_159

    # Use 3 folds for the nine-hour MVP and retain 5 folds for the
    # post-competition GitHub version without regenerating the dataset.
    CV_SPLITS: tuple[int, ...] = (3, 5)

    CANONICAL_AMINO_ACIDS: tuple[str, ...] = tuple(
        "ACDEFGHIKLMNPQRSTVWY"
    )

    # These columns contain the measured outcome, uncertainty derived from
    # that outcome, or post-experimental quality information. They must not
    # become model inputs.
    LEAKAGE_COLUMNS: tuple[str, ...] = (
        "score",
        "se",
        "LLR",
        "LLR_ci_lower",
        "LLR_ci_upper",
        "LLR_evidence_strength",
        "eligible_simple_missense",
        "ref_match",
        "reference_position_valid",
        "reference_residue",
    )


# ---------------------------------------------------------------------
# Reproducible amino-acid property tables
# ---------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity scale.
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Approximate amino-acid molecular weights in daltons.
MOLECULAR_WEIGHT = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10,
    "C": 121.16, "Q": 146.15, "E": 147.13, "G": 75.07,
    "H": 155.16, "I": 131.18, "L": 131.18, "K": 146.19,
    "M": 149.21, "F": 165.19, "P": 115.13, "S": 105.09,
    "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

# Common residue-volume scale in cubic angstroms.
RESIDUE_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
    "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}

# Grantham polarity values. These are used as continuous properties,
# not as a precomputed Grantham substitution-distance matrix.
POLARITY = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5,
    "Q": 10.5, "E": 12.3, "G": 9.0, "H": 10.4, "I": 5.2,
    "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2, "P": 8.0,
    "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
}

# Simplified side-chain charge at approximately physiological pH.
CHARGE_VALUE = {
    "D": -1, "E": -1, "K": 1, "R": 1, "H": 1,
    "A": 0, "C": 0, "F": 0, "G": 0, "I": 0,
    "L": 0, "M": 0, "N": 0, "P": 0, "Q": 0,
    "S": 0, "T": 0, "V": 0, "W": 0, "Y": 0,
}

AROMATIC = set("FWY")
ALIPHATIC = set("AILMV")
HYDROXYL = set("STY")
SULFUR_CONTAINING = set("CM")
AMIDE = set("NQ")
ACIDIC = set("DE")
BASIC = set("HKR")


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def calculate_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 checksum for one file."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write formatted JSON with NumPy-safe scalar conversion."""

    def default_converter(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable."
        )

    output_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=default_converter,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_property_tables() -> None:
    """Confirm that every canonical amino acid has every property."""
    expected = set(CFG.CANONICAL_AMINO_ACIDS)
    tables = {
        "HYDROPHOBICITY": HYDROPHOBICITY,
        "MOLECULAR_WEIGHT": MOLECULAR_WEIGHT,
        "RESIDUE_VOLUME": RESIDUE_VOLUME,
        "POLARITY": POLARITY,
        "CHARGE_VALUE": CHARGE_VALUE,
    }

    for table_name, table in tables.items():
        observed = set(table)
        if observed != expected:
            raise ValueError(
                f"{table_name} amino-acid mismatch. "
                f"Missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def load_and_validate_input() -> pd.DataFrame:
    """Load the validated missense-domain dataset and verify invariants."""
    if not CFG.INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing modeling input: {CFG.INPUT_PATH}\n"
            "Run Section 14.7 of the EDA first."
        )

    dataframe = pd.read_parquet(CFG.INPUT_PATH)

    required_columns = {
        "variant_id",
        "hgvs_pro_normalized",
        "wt_aa1",
        "mut_aa1",
        "position",
        CFG.TARGET_COLUMN,
        "variant_class",
        "ref_match",
        "protein_variant_duplicate",
    }

    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "The input dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) != CFG.EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected number of rows: {len(dataframe):,}. "
            f"Expected {CFG.EXPECTED_ROWS:,}."
        )

    dataframe["position"] = dataframe["position"].astype("int16")

    if not dataframe["variant_class"].eq("simple_missense").all():
        raise ValueError("The input contains non-missense variants.")
    if not dataframe["ref_match"].eq(True).all():
        raise ValueError("Some variants do not match NP_000229.1.")
    if dataframe["protein_variant_duplicate"].any():
        raise ValueError("Duplicated protein variants were detected.")
    if dataframe[CFG.TARGET_COLUMN].isna().any():
        raise ValueError("The target contains missing values.")
    if dataframe["variant_id"].duplicated().any():
        raise ValueError("variant_id is not unique.")

    observed_amino_acids = set(dataframe["wt_aa1"]) | set(
        dataframe["mut_aa1"]
    )
    unexpected_amino_acids = observed_amino_acids - set(
        CFG.CANONICAL_AMINO_ACIDS
    )
    if unexpected_amino_acids:
        raise ValueError(
            "Unexpected amino acids were found: "
            f"{sorted(unexpected_amino_acids)}"
        )

    if dataframe["wt_aa1"].eq(dataframe["mut_aa1"]).any():
        raise ValueError("The missense dataset contains unchanged residues.")

    if not dataframe["position"].between(
        1, CFG.EXPECTED_PROTEIN_LENGTH
    ).all():
        raise ValueError("One or more positions are outside NP_000229.1.")

    domain_columns = [
        column
        for column in dataframe.columns
        if column.startswith("domain__")
    ]
    if not domain_columns:
        raise ValueError(
            "No domain__ columns were found. Run Section 14.7 of the EDA."
        )

    for column in domain_columns:
        invalid_values = set(dataframe[column].dropna().unique()) - {
            0, 1, False, True
        }
        if invalid_values:
            raise ValueError(
                f"Domain feature {column} contains non-binary values: "
                f"{invalid_values}"
            )

    return dataframe


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def add_amino_acid_properties(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Add WT, mutant, delta, and absolute-delta properties."""
    dataframe = dataframe.copy()
    property_tables = {
        "hydrophobicity": HYDROPHOBICITY,
        "molecular_weight": MOLECULAR_WEIGHT,
        "residue_volume": RESIDUE_VOLUME,
        "polarity": POLARITY,
        "charge": CHARGE_VALUE,
    }
    generated_columns: list[str] = []

    for property_name, property_map in property_tables.items():
        wt_column = f"wt_{property_name}"
        mut_column = f"mut_{property_name}"
        delta_column = f"delta_{property_name}"
        absolute_column = f"abs_delta_{property_name}"

        dataframe[wt_column] = (
            dataframe["wt_aa1"].map(property_map).astype("float32")
        )
        dataframe[mut_column] = (
            dataframe["mut_aa1"].map(property_map).astype("float32")
        )
        dataframe[delta_column] = (
            dataframe[mut_column] - dataframe[wt_column]
        ).astype("float32")
        dataframe[absolute_column] = (
            dataframe[delta_column].abs().astype("float32")
        )

        generated_columns.extend(
            [wt_column, mut_column, delta_column, absolute_column]
        )

    return dataframe, generated_columns


def add_binary_residue_properties(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Add interpretable binary properties for WT and mutant residues."""
    dataframe = dataframe.copy()
    residue_sets = {
        "aromatic": AROMATIC,
        "aliphatic": ALIPHATIC,
        "hydroxyl": HYDROXYL,
        "sulfur_containing": SULFUR_CONTAINING,
        "amide": AMIDE,
        "acidic": ACIDIC,
        "basic": BASIC,
        "proline": {"P"},
        "glycine": {"G"},
        "cysteine": {"C"},
    }
    generated_columns: list[str] = []

    for property_name, amino_acid_set in residue_sets.items():
        wt_column = f"wt_is_{property_name}"
        mut_column = f"mut_is_{property_name}"
        introduces_column = f"introduces_{property_name}"
        removes_column = f"removes_{property_name}"

        dataframe[wt_column] = (
            dataframe["wt_aa1"].isin(amino_acid_set).astype("int8")
        )
        dataframe[mut_column] = (
            dataframe["mut_aa1"].isin(amino_acid_set).astype("int8")
        )
        dataframe[introduces_column] = (
            dataframe[wt_column].eq(0) & dataframe[mut_column].eq(1)
        ).astype("int8")
        dataframe[removes_column] = (
            dataframe[wt_column].eq(1) & dataframe[mut_column].eq(0)
        ).astype("int8")

        generated_columns.extend(
            [wt_column, mut_column, introduces_column, removes_column]
        )

    dataframe["same_charge_class"] = (
        dataframe["wt_charge"].eq(dataframe["mut_charge"]).astype("int8")
    )
    dataframe["introduces_charge"] = (
        dataframe["wt_charge"].eq(0) & dataframe["mut_charge"].ne(0)
    ).astype("int8")
    dataframe["removes_charge"] = (
        dataframe["wt_charge"].ne(0) & dataframe["mut_charge"].eq(0)
    ).astype("int8")
    dataframe["charge_reversal"] = (
        dataframe["wt_charge"].mul(dataframe["mut_charge"]).eq(-1)
    ).astype("int8")

    generated_columns.extend(
        [
            "same_charge_class",
            "introduces_charge",
            "removes_charge",
            "charge_reversal",
        ]
    )

    return dataframe, generated_columns


def add_substitution_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Add BLOSUM62, positional, distance, and one-hot features."""
    dataframe = dataframe.copy()
    generated_columns: list[str] = []
    blosum62 = substitution_matrices.load("BLOSUM62")

    def get_blosum_score(wt_amino_acid: str, mutant_amino_acid: str) -> float:
        try:
            return float(blosum62[wt_amino_acid, mutant_amino_acid])
        except (IndexError, KeyError):
            return float(blosum62[mutant_amino_acid, wt_amino_acid])

    dataframe["blosum62"] = np.asarray(
        [
            get_blosum_score(wt, mutant)
            for wt, mutant in zip(dataframe["wt_aa1"], dataframe["mut_aa1"])
        ],
        dtype=np.float32,
    )

    dataframe["relative_position"] = (
        dataframe["position"] / CFG.EXPECTED_PROTEIN_LENGTH
    ).astype("float32")
    dataframe["distance_to_n_terminus"] = (
        dataframe["position"] - 1
    ).astype("int16")
    dataframe["distance_to_c_terminus"] = (
        CFG.EXPECTED_PROTEIN_LENGTH - dataframe["position"]
    ).astype("int16")
    dataframe["relative_terminal_distance"] = (
        np.minimum(
            dataframe["distance_to_n_terminus"],
            dataframe["distance_to_c_terminus"],
        )
        / CFG.EXPECTED_PROTEIN_LENGTH
    ).astype("float32")

    # Standardized Euclidean distance across four physicochemical
    # properties. This is an interpretable severity measure, not the
    # official Grantham substitution-distance matrix.
    property_matrix = pd.DataFrame(
        {
            "hydrophobicity": pd.Series(HYDROPHOBICITY),
            "molecular_weight": pd.Series(MOLECULAR_WEIGHT),
            "residue_volume": pd.Series(RESIDUE_VOLUME),
            "polarity": pd.Series(POLARITY),
        }
    )
    property_stds = property_matrix.std(ddof=0)
    standardized_squared_deltas = [
        (
            dataframe[f"delta_{property_name}"]
            / property_stds[property_name]
        )
        .pow(2)
        .to_numpy()
        for property_name in property_matrix.columns
    ]
    dataframe["physicochemical_distance"] = np.sqrt(
        np.sum(standardized_squared_deltas, axis=0)
    ).astype("float32")

    generated_columns.extend(
        [
            "blosum62",
            "relative_position",
            "distance_to_n_terminus",
            "distance_to_c_terminus",
            "relative_terminal_distance",
            "physicochemical_distance",
        ]
    )

    # Fixed one-hot encoding guarantees the same schema locally and on
    # Kaggle, even when one validation fold lacks a category.
    for amino_acid in CFG.CANONICAL_AMINO_ACIDS:
        wt_column = f"wt_aa_{amino_acid}"
        mut_column = f"mut_aa_{amino_acid}"
        dataframe[wt_column] = (
            dataframe["wt_aa1"].eq(amino_acid).astype("int8")
        )
        dataframe[mut_column] = (
            dataframe["mut_aa1"].eq(amino_acid).astype("int8")
        )
        generated_columns.extend([wt_column, mut_column])

    return dataframe, generated_columns


def add_group_folds(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Create reusable GroupKFold assignments by protein position."""
    dataframe = dataframe.copy()
    fold_columns: list[str] = []

    for number_of_splits in CFG.CV_SPLITS:
        fold_column = f"fold_{number_of_splits}"
        dataframe[fold_column] = np.int8(-1)
        splitter = GroupKFold(n_splits=number_of_splits)

        for fold_index, (_, validation_indices) in enumerate(
            splitter.split(
                X=dataframe,
                y=dataframe[CFG.TARGET_COLUMN],
                groups=dataframe[CFG.GROUP_COLUMN],
            )
        ):
            dataframe.loc[
                dataframe.index[validation_indices], fold_column
            ] = np.int8(fold_index)

        if dataframe[fold_column].lt(0).any():
            raise ValueError(f"Incomplete fold assignment: {fold_column}")

        folds_per_position = dataframe.groupby(CFG.GROUP_COLUMN)[
            fold_column
        ].nunique()
        if not folds_per_position.eq(1).all():
            raise ValueError(f"Position leakage detected in {fold_column}.")

        fold_columns.append(fold_column)

    return dataframe, fold_columns


# ---------------------------------------------------------------------
# Output construction
# ---------------------------------------------------------------------

def build_modeling_dataset() -> None:
    """Build and save the model-ready, leakage-safe feature dataset."""
    validate_property_tables()
    dataframe = load_and_validate_input()

    domain_feature_columns = sorted(
        [column for column in dataframe.columns if column.startswith("domain__")]
    )

    dataframe, continuous_property_columns = add_amino_acid_properties(
        dataframe
    )
    dataframe, binary_property_columns = add_binary_residue_properties(
        dataframe
    )
    dataframe, substitution_feature_columns = add_substitution_features(
        dataframe
    )
    dataframe, fold_columns = add_group_folds(dataframe)

    dataframe[domain_feature_columns] = (
        dataframe[domain_feature_columns].fillna(0).astype("int8")
    )

    biochemical_feature_columns = list(
        dict.fromkeys(
            continuous_property_columns
            + binary_property_columns
            + substitution_feature_columns
        )
    )
    model_2_feature_columns = (
        biochemical_feature_columns + domain_feature_columns
    )

    metadata_columns = [
        "variant_id",
        "hgvs_pro_normalized",
        "wt_aa1",
        "mut_aa1",
        "position",
    ]
    optional_metadata_columns = [
        column
        for column in ["accession", "hgvs_pro"]
        if column in dataframe.columns
    ]

    output_columns = list(
        dict.fromkeys(
            metadata_columns
            + optional_metadata_columns
            + fold_columns
            + [CFG.TARGET_COLUMN]
            + model_2_feature_columns
        )
    )
    modeling_dataframe = dataframe[output_columns].copy()
    modeling_dataframe[CFG.TARGET_COLUMN] = modeling_dataframe[
        CFG.TARGET_COLUMN
    ].astype("float32")

    forbidden_present = set(CFG.LEAKAGE_COLUMNS) & set(
        modeling_dataframe.columns
    )
    if forbidden_present:
        raise ValueError(
            "Leakage columns remained in the modeling dataset: "
            f"{sorted(forbidden_present)}"
        )

    feature_missing_counts = modeling_dataframe[
        model_2_feature_columns
    ].isna().sum()
    features_with_missing = feature_missing_counts.loc[
        feature_missing_counts.gt(0)
    ]
    if not features_with_missing.empty:
        raise ValueError(
            "Model features contain missing values:\n"
            f"{features_with_missing}"
        )

    if modeling_dataframe["variant_id"].duplicated().any():
        raise ValueError("variant_id is duplicated in the output.")

    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    modeling_dataframe.to_parquet(
        CFG.OUTPUT_PARQUET_PATH,
        index=False,
        compression="snappy",
    )
    modeling_dataframe.to_csv(
        CFG.OUTPUT_CSV_PATH,
        index=False,
        compression="gzip",
    )

    feature_schema = {
        "dataset_name": "CardioVUS KCNH2 model-ready features",
        "target_column": CFG.TARGET_COLUMN,
        "group_column": CFG.GROUP_COLUMN,
        "fold_columns": fold_columns,
        "metadata_columns": metadata_columns + optional_metadata_columns,
        "model_0": {
            "name": "constant_median_baseline",
            "feature_columns": [],
            "description": (
                "Predicts the training-fold median. No predictive features."
            ),
        },
        "model_1": {
            "name": "xgboost_biochemical",
            "feature_columns": biochemical_feature_columns,
            "description": (
                "Biochemical, substitution, amino-acid identity, and "
                "positional features."
            ),
        },
        "model_2": {
            "name": "xgboost_biochemical_domains",
            "feature_columns": model_2_feature_columns,
            "description": (
                "Model 1 features plus curated UniProt domain and "
                "topology indicators."
            ),
        },
        "future_model_3": {
            "name": "xgboost_biochemical_domains_esm2",
            "merge_keys": ["position"],
            "description": (
                "Model 2 features plus one WT contextual ESM-2 embedding "
                "per protein position."
            ),
        },
        "biochemical_feature_columns": biochemical_feature_columns,
        "domain_feature_columns": domain_feature_columns,
        "excluded_experimental_columns": list(CFG.LEAKAGE_COLUMNS),
    }
    write_json(feature_schema, CFG.FEATURE_SCHEMA_PATH)

    fold_distributions = {
        fold_column: {
            str(fold): int(count)
            for fold, count in modeling_dataframe[fold_column]
            .value_counts(sort=False)
            .sort_index()
            .items()
        }
        for fold_column in fold_columns
    }

    dataset_summary = {
        "input_path": str(CFG.INPUT_PATH.relative_to(CFG.ROOT_DIR)),
        "output_parquet_path": str(
            CFG.OUTPUT_PARQUET_PATH.relative_to(CFG.ROOT_DIR)
        ),
        "output_csv_path": str(
            CFG.OUTPUT_CSV_PATH.relative_to(CFG.ROOT_DIR)
        ),
        "rows": int(len(modeling_dataframe)),
        "columns": int(modeling_dataframe.shape[1]),
        "unique_variants": int(modeling_dataframe["variant_id"].nunique()),
        "unique_positions": int(modeling_dataframe["position"].nunique()),
        "biochemical_feature_count": int(
            len(biochemical_feature_columns)
        ),
        "domain_feature_count": int(len(domain_feature_columns)),
        "model_2_feature_count": int(len(model_2_feature_columns)),
        "target_summary": {
            str(key): float(value)
            for key, value in modeling_dataframe[CFG.TARGET_COLUMN]
            .describe(
                percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
            )
            .items()
        },
        "fold_distributions": fold_distributions,
        "parquet_sha256": calculate_sha256(CFG.OUTPUT_PARQUET_PATH),
        "csv_gz_sha256": calculate_sha256(CFG.OUTPUT_CSV_PATH),
    }
    write_json(dataset_summary, CFG.DATASET_SUMMARY_PATH)

    print("=" * 76)
    print("KCNH2 modeling dataset generated successfully")
    print("=" * 76)
    print(f"Rows: {len(modeling_dataframe):,}")
    print(
        "Unique positions: "
        f"{modeling_dataframe['position'].nunique():,}"
    )
    print(
        "Biochemical features: "
        f"{len(biochemical_feature_columns):,}"
    )
    print(f"Domain features: {len(domain_feature_columns):,}")
    print(
        "Model 2 total features: "
        f"{len(model_2_feature_columns):,}"
    )
    print("\nFold distributions:")
    for fold_column, distribution in fold_distributions.items():
        print(f"  {fold_column}: {distribution}")
    print("\nOutputs:")
    print(f"  Parquet: {CFG.OUTPUT_PARQUET_PATH}")
    print(f"  CSV.gz:  {CFG.OUTPUT_CSV_PATH}")
    print(f"  Schema:  {CFG.FEATURE_SCHEMA_PATH}")
    print(f"  Summary: {CFG.DATASET_SUMMARY_PATH}")


def main() -> int:
    """Run the feature-generation workflow."""
    try:
        build_modeling_dataset()
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        KeyError,
    ) as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
