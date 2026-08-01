# %% [markdown]
## 14. Protein-domain analysis — purpose: determine whether trafficking defects cluster in specific structural and topological regions

This section integrates curated UniProtKB annotations with the MaveDB missense map.

Biological interpretation:

- **Domain:** a region with a defined structural or functional role.
- **Transmembrane segment:** a hydrophobic helix crossing the lipid membrane.
- **Topological domain:** a protein segment located on the cytoplasmic or extracellular side.
- **Region or motif:** a sequence interval with an annotated functional property.

The analysis asks whether variants inside each feature have systematically lower trafficking scores than variants outside it. Features may overlap, so every feature is analyzed independently rather than forcing each residue into one exclusive category.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from IPython.display import Markdown, display


FEATURES_PATH = (
    ROOT_DIR
    / "data"
    / "external"
    / "kcnh2_uniprot_features.csv"
)

DOMAINS_PATH = (
    ROOT_DIR
    / "data"
    / "external"
    / "kcnh2_domains.csv"
)

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        "Missing UniProt feature table. Run:\n"
        "python src/cardiovus/data/"
        "download_uniprot_features.py"
    )

features_df = pd.read_csv(FEATURES_PATH)
domains_df = pd.read_csv(DOMAINS_PATH)

required_columns = {
    "feature_label",
    "feature_type",
    "feature_group",
    "start",
    "end",
    "length",
}

missing_columns = (
    required_columns
    - set(features_df.columns)
)

if missing_columns:
    raise ValueError(
        "Missing annotation columns: "
        f"{sorted(missing_columns)}"
    )

assert features_df["start"].ge(1).all()
assert features_df["end"].le(
    len(reference_sequence)
).all()
assert features_df["start"].le(
    features_df["end"]
).all()

print("Feature counts by annotation layer:")
display(
    features_df[
        "feature_group"
    ]
    .value_counts()
    .rename_axis("feature_group")
    .reset_index(name="count")
)

display(
    features_df[
        [
            "feature_label",
            "feature_type",
            "feature_group",
            "start",
            "end",
            "length",
            "evidence_codes",
        ]
    ]
)

# %% [markdown]
### 14.1 Functional categories — purpose: summarize trafficking behavior without converting it into a clinical diagnosis

The associated MAVE study defines approximately 58–153% of wild-type trafficking as the normal functional range. Scores below 58 indicate reduced trafficking, and scores below 35 represent severe loss of trafficking in the assay calibration.

These categories describe the molecular assay only. They are not final ACMG/AMP classifications.

# %%
LOW_TRAFFICKING_THRESHOLD = 58.0
SEVERE_LOSS_THRESHOLD = 35.0
WT_LIKE_UPPER_THRESHOLD = 153.0


def assign_trafficking_category(
    score: float,
) -> str:
    if score < SEVERE_LOSS_THRESHOLD:
        return "severe_loss_lt_35"

    if score < LOW_TRAFFICKING_THRESHOLD:
        return "reduced_35_to_58"

    if score <= WT_LIKE_UPPER_THRESHOLD:
        return "wt_like_58_to_153"

    return "above_wt_gt_153"


domain_variant_base = missense_df.copy()

domain_variant_base[
    "trafficking_category"
] = domain_variant_base[
    "score_numeric"
].apply(assign_trafficking_category)

display(
    domain_variant_base[
        "trafficking_category"
    ]
    .value_counts()
    .rename_axis("category")
    .reset_index(name="count")
)

# %% [markdown]
### 14.2 Position-level annotation — purpose: map every residue to all overlapping UniProt features

A residue can belong to more than one annotation. For example, it may lie inside a broad ion-channel region and also inside a transmembrane helix. We therefore create a **long-format membership table** rather than assigning a single domain.

# %%
analysis_groups = {
    "domain",
    "region",
    "membrane",
    "topology",
}

analysis_features = features_df.loc[
    features_df[
        "feature_group"
    ].isin(analysis_groups)
].copy()

membership_rows = []

for feature in analysis_features.itertuples(
    index=False
):
    for position in range(
        int(feature.start),
        int(feature.end) + 1,
    ):
        membership_rows.append(
            {
                "position": position,
                "feature_label": (
                    feature.feature_label
                ),
                "feature_type": (
                    feature.feature_type
                ),
                "feature_group": (
                    feature.feature_group
                ),
                "feature_start": int(
                    feature.start
                ),
                "feature_end": int(
                    feature.end
                ),
            }
        )

position_feature_membership = pd.DataFrame(
    membership_rows
)

print(
    "Annotated residue-feature memberships:",
    f"{len(position_feature_membership):,}",
)

print(
    "Unique annotated positions:",
    position_feature_membership[
        "position"
    ].nunique(),
)

display(
    position_feature_membership.head()
)

position_feature_membership.to_csv(
    EDA_REPORTS_DIR
    / "kcnh2_position_feature_membership.csv",
    index=False,
)

# %% [markdown]
### 14.3 Domain-level summary — purpose: quantify coverage, central tendency, variability, and loss-of-trafficking fractions

Two levels are reported:

1. **Variant level:** all amino acid substitutions inside the feature.
2. **Position level:** the median score across substitutions at each residue.

The position-level summary is more appropriate for statistical comparison because substitutions at the same residue share structural context and are not fully independent.

# %%
position_score_summary = (
    domain_variant_base.groupby("position")
    .agg(
        wt_residue=("wt_aa1", "first"),
        n_variants=("variant_id", "count"),
        median_score=(
            "score_numeric",
            "median",
        ),
        mean_score=(
            "score_numeric",
            "mean",
        ),
        q1_score=(
            "score_numeric",
            lambda values: values.quantile(
                0.25
            ),
        ),
        q3_score=(
            "score_numeric",
            lambda values: values.quantile(
                0.75
            ),
        ),
    )
    .reset_index()
)

position_score_summary["iqr_score"] = (
    position_score_summary["q3_score"]
    - position_score_summary["q1_score"]
)


def rank_biserial_from_u(
    u_statistic: float,
    n_inside: int,
    n_outside: int,
) -> float:
    return (
        2
        * u_statistic
        / (n_inside * n_outside)
        - 1
    )


def holm_adjust(
    p_values: pd.Series,
) -> np.ndarray:
    values = p_values.to_numpy(
        dtype=float
    )
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    number_of_tests = len(values)

    for rank, original_index in enumerate(
        order
    ):
        candidate = (
            number_of_tests - rank
        ) * values[original_index]

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[original_index] = min(
            running_max,
            1.0,
        )

    return adjusted


summary_rows = []

for feature in analysis_features.itertuples(
    index=False
):
    inside_variant_mask = (
        domain_variant_base[
            "position"
        ].between(
            int(feature.start),
            int(feature.end),
        )
    )

    inside_variants = (
        domain_variant_base.loc[
            inside_variant_mask
        ]
    )
    outside_variants = (
        domain_variant_base.loc[
            ~inside_variant_mask
        ]
    )

    inside_positions = (
        position_score_summary.loc[
            position_score_summary[
                "position"
            ].between(
                int(feature.start),
                int(feature.end),
            )
        ]
    )
    outside_positions = (
        position_score_summary.loc[
            ~position_score_summary[
                "position"
            ].between(
                int(feature.start),
                int(feature.end),
            )
        ]
    )

    if (
        len(inside_positions) >= 5
        and len(outside_positions) >= 5
    ):
        test = stats.mannwhitneyu(
            inside_positions["median_score"],
            outside_positions["median_score"],
            alternative="two-sided",
        )

        effect_size = (
            rank_biserial_from_u(
                test.statistic,
                len(inside_positions),
                len(outside_positions),
            )
        )
        p_value = test.pvalue
        u_statistic = test.statistic

    else:
        p_value = np.nan
        u_statistic = np.nan
        effect_size = np.nan

    summary_rows.append(
        {
            "feature_label": (
                feature.feature_label
            ),
            "feature_type": (
                feature.feature_type
            ),
            "feature_group": (
                feature.feature_group
            ),
            "start": int(feature.start),
            "end": int(feature.end),
            "feature_length": int(
                feature.length
            ),
            "n_observed_positions": (
                inside_positions[
                    "position"
                ].nunique()
            ),
            "n_variants": len(
                inside_variants
            ),
            "variant_mean_score": (
                inside_variants[
                    "score_numeric"
                ].mean()
            ),
            "variant_median_score": (
                inside_variants[
                    "score_numeric"
                ].median()
            ),
            "position_median_score": (
                inside_positions[
                    "median_score"
                ].median()
            ),
            "position_iqr": (
                inside_positions[
                    "median_score"
                ].quantile(0.75)
                - inside_positions[
                    "median_score"
                ].quantile(0.25)
            ),
            "severe_loss_fraction_lt_35": (
                inside_variants[
                    "score_numeric"
                ].lt(
                    SEVERE_LOSS_THRESHOLD
                ).mean()
            ),
            "reduced_fraction_lt_58": (
                inside_variants[
                    "score_numeric"
                ].lt(
                    LOW_TRAFFICKING_THRESHOLD
                ).mean()
            ),
            "wt_like_fraction_58_153": (
                inside_variants[
                    "score_numeric"
                ].between(
                    LOW_TRAFFICKING_THRESHOLD,
                    WT_LIKE_UPPER_THRESHOLD,
                    inclusive="both",
                ).mean()
            ),
            "above_wt_fraction_gt_153": (
                inside_variants[
                    "score_numeric"
                ].gt(
                    WT_LIKE_UPPER_THRESHOLD
                ).mean()
            ),
            "mannwhitney_u_position_level": (
                u_statistic
            ),
            "p_value_position_level": (
                p_value
            ),
            "rank_biserial_effect": (
                effect_size
            ),
        }
    )

domain_summary_df = pd.DataFrame(
    summary_rows
)

valid_tests = domain_summary_df[
    "p_value_position_level"
].notna()

domain_summary_df[
    "p_holm_position_level"
] = np.nan

domain_summary_df.loc[
    valid_tests,
    "p_holm_position_level",
] = holm_adjust(
    domain_summary_df.loc[
        valid_tests,
        "p_value_position_level",
    ]
)

domain_summary_df[
    "significantly_different_after_holm"
] = (
    domain_summary_df[
        "p_holm_position_level"
    ].lt(0.05)
)

display(
    domain_summary_df.sort_values(
        "position_median_score"
    )
)

domain_summary_df.to_csv(
    EDA_REPORTS_DIR
    / "kcnh2_domain_functional_summary.csv",
    index=False,
)

# %% [markdown]
### 14.4 Visualization — purpose: identify feature-level intolerance and its magnitude

A negative rank-biserial effect indicates that residue-level median scores inside the feature tend to be lower than outside it. Statistical significance should be interpreted together with effect size and biological plausibility.

# %%
plot_summary = (
    domain_summary_df.loc[
        domain_summary_df[
            "n_observed_positions"
        ].ge(5)
    ]
    .sort_values(
        "position_median_score"
    )
)

plt.figure(
    figsize=(
        11,
        max(
            6,
            0.38 * len(plot_summary),
        ),
    )
)

plt.barh(
    plot_summary["feature_label"],
    plot_summary[
        "position_median_score"
    ],
)

plt.axvline(
    LOW_TRAFFICKING_THRESHOLD,
    linestyle="--",
    linewidth=1,
    label="Reduced-trafficking threshold (58)",
)
plt.axvline(
    100,
    linestyle=":",
    linewidth=1,
    label="Approximate WT reference (100)",
)
plt.axvline(
    WT_LIKE_UPPER_THRESHOLD,
    linestyle="--",
    linewidth=1,
    label="Upper WT-like threshold (153)",
)

plt.xlabel(
    "Median of residue-level median trafficking scores"
)
plt.ylabel("UniProt feature")
plt.title(
    "KCNH2 functional tolerance by annotated feature\n"
    "Purpose: identify regions enriched in reduced trafficking"
)
plt.legend()
plt.tight_layout()
plt.show()

# %%
plt.figure(
    figsize=(
        11,
        max(
            6,
            0.38 * len(plot_summary),
        ),
    )
)

plt.barh(
    plot_summary["feature_label"],
    plot_summary[
        "reduced_fraction_lt_58"
    ],
)

plt.xlabel(
    "Fraction of missense variants with score < 58"
)
plt.ylabel("UniProt feature")
plt.title(
    "Reduced-trafficking burden by annotated feature\n"
    "Purpose: compare the proportion of functionally impaired variants"
)
plt.tight_layout()
plt.show()

# %%
plt.figure(
    figsize=(
        11,
        max(
            6,
            0.38 * len(plot_summary),
        ),
    )
)

plt.barh(
    plot_summary["feature_label"],
    plot_summary[
        "rank_biserial_effect"
    ],
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1,
)

plt.xlabel(
    "Rank-biserial effect: inside feature vs outside"
)
plt.ylabel("UniProt feature")
plt.title(
    "Position-level effect size by annotated feature\n"
    "Negative values indicate lower trafficking inside the feature"
)
plt.tight_layout()
plt.show()

# %% [markdown]
### 14.5 Sequence-level view — purpose: connect domain boundaries with residue-specific sensitivity

This plot shows residue-level median scores across the 1,159-amino-acid sequence. It should be interpreted together with the feature table because UniProt annotations can overlap.

# %%
plt.figure(figsize=(15, 6))

plt.plot(
    position_score_summary["position"],
    position_score_summary[
        "median_score"
    ],
    linewidth=0.8,
)

plt.axhline(
    LOW_TRAFFICKING_THRESHOLD,
    linestyle="--",
    linewidth=1,
    label="Reduced trafficking (58)",
)
plt.axhline(
    100,
    linestyle=":",
    linewidth=1,
    label="WT reference (100)",
)
plt.axhline(
    WT_LIKE_UPPER_THRESHOLD,
    linestyle="--",
    linewidth=1,
    label="Upper WT-like boundary (153)",
)

plt.xlabel("Position in NP_000229.1")
plt.ylabel(
    "Median trafficking score across substitutions"
)
plt.title(
    "Residue-level KCNH2 mutational sensitivity\n"
    "Purpose: locate continuous clusters of trafficking intolerance"
)
plt.legend()
plt.tight_layout()
plt.show()

# %%
track_features = analysis_features.loc[
    analysis_features[
        "feature_group"
    ].isin(
        {
            "domain",
            "region",
            "membrane",
        }
    )
].sort_values(
    ["start", "end"]
).reset_index(drop=True)

plt.figure(
    figsize=(
        15,
        max(
            5,
            0.45 * len(track_features),
        ),
    )
)

for row_index, feature in (
    track_features.iterrows()
):
    plt.barh(
        y=row_index,
        width=(
            int(feature["end"])
            - int(feature["start"])
            + 1
        ),
        left=int(feature["start"]),
        height=0.7,
    )

plt.yticks(
    ticks=np.arange(
        len(track_features)
    ),
    labels=track_features[
        "feature_label"
    ],
)

plt.xlabel("Position in NP_000229.1")
plt.ylabel("UniProt feature")
plt.title(
    "Curated KCNH2 domain and membrane-feature map\n"
    "Purpose: document the structural context used in the EDA"
)
plt.tight_layout()
plt.show()

# %% [markdown]
### 14.6 Most intolerant residues within each feature — purpose: identify local hotspots that drive domain-level effects

A domain may have a low overall score because many residues are moderately sensitive or because a small cluster is extremely intolerant. Listing the lowest-scoring residues helps distinguish these patterns.

# %%
intolerant_position_rows = []

for feature in analysis_features.itertuples(
    index=False
):
    feature_positions = (
        position_score_summary.loc[
            position_score_summary[
                "position"
            ].between(
                int(feature.start),
                int(feature.end),
            )
        ]
        .sort_values(
            "median_score"
        )
        .head(10)
        .copy()
    )

    if feature_positions.empty:
        continue

    feature_positions.insert(
        0,
        "feature_label",
        feature.feature_label,
    )
    feature_positions.insert(
        1,
        "feature_group",
        feature.feature_group,
    )

    intolerant_position_rows.append(
        feature_positions
    )

if intolerant_position_rows:
    intolerant_positions_df = pd.concat(
        intolerant_position_rows,
        ignore_index=True,
    )
else:
    intolerant_positions_df = pd.DataFrame()

display(
    intolerant_positions_df.head(50)
)

intolerant_positions_df.to_csv(
    EDA_REPORTS_DIR
    / "kcnh2_intolerant_positions_by_feature.csv",
    index=False,
)

# %% [markdown]
### 14.7 Modeling dataset — purpose: convert biological annotations into reproducible machine-learning features

Each variant receives one binary column per UniProt feature. Overlapping annotations are preserved. These columns can be used by the biochemical baseline and the combined ESM-2 model.

# %%
position_feature_matrix = (
    position_feature_membership.assign(
        present=1
    )
    .pivot_table(
        index="position",
        columns="feature_label",
        values="present",
        aggfunc="max",
        fill_value=0,
    )
    .reset_index()
)

position_feature_matrix.columns = [
    (
        "domain__"
        + str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )
    if column != "position"
    else "position"
    for column in position_feature_matrix.columns
]

missense_with_domains = (
    domain_variant_base.merge(
        position_feature_matrix,
        on="position",
        how="left",
        validate="many_to_one",
    )
)

domain_feature_columns = [
    column
    for column in missense_with_domains.columns
    if column.startswith("domain__")
]

missense_with_domains[
    domain_feature_columns
] = missense_with_domains[
    domain_feature_columns
].fillna(0).astype("int8")

DOMAIN_DATASET_PATH = (
    INTERIM_DIR
    / "kcnh2_variants_with_domains.parquet"
)

missense_with_domains.to_parquet(
    DOMAIN_DATASET_PATH,
    index=False,
)

print(
    f"Domain feature columns: "
    f"{len(domain_feature_columns)}"
)
print(
    f"Output: {DOMAIN_DATASET_PATH}"
)

assert len(
    missense_with_domains
) == len(domain_variant_base)

# %% [markdown]
### 14.8 Domain-analysis conclusions — purpose: document biological interpretation before model training

After running the cells, summarize:

1. Which domains or membrane regions have the lowest residue-level median scores?
2. Which features have the highest fraction of variants below 58?
3. Which differences remain significant after Holm correction?
4. Are negative effects broad across the feature or driven by a small hotspot?
5. Do transmembrane or pore-associated regions show greater intolerance than long terminal regions?
6. Which domain annotations should be included as model features?
7. Which findings concern trafficking only and should not be generalized to complete channel function?

Add the answers to Sections 17.3, 17.7, and 17.8 of the EDA conclusions.
