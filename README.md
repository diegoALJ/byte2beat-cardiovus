# CardioVUS-KCNH2

**Functional prediction and evidence-grounded interpretation of KCNH2 missense variants**

CardioVUS-KCNH2 is a bioinformatics research project developed for the Byte 2 Beat Hackathon. The project investigates whether biochemical properties and protein language model representations can predict the functional effects of missense variants in **KCNH2**, a gene strongly associated with Long QT Syndrome type 2.

## Research question

Can a machine learning model combining biochemical features and ESM-2 protein embeddings predict the experimentally measured functional effects of previously unseen KCNH2 missense variants?

## Project objectives

* Curate functional variant measurements from MaveDB.
* Validate variants against a canonical KCNH2 reference sequence.
* Generate biochemical and sequence-based features.
* Represent wild-type and mutant sequences using ESM-2.
* Train XGBoost models to predict experimental functional scores.
* Evaluate generalization using residue-grouped cross-validation.
* Compare model predictions with ClinVar and AlphaMissense evidence.
* Generate evidence-grounded variant reports through a retrieval-augmented generation pipeline.

## Intended use

This project is intended for research and educational purposes. It aims to prioritize variants for further investigation and summarize available evidence.

It does not provide clinical diagnoses, formal ACMG/AMP classifications, or treatment recommendations.

## Planned data sources

* MaveDB — experimental multiplexed assay scores.
* NCBI RefSeq — canonical KCNH2 protein sequence.
* ClinVar — clinical variant interpretations.
* CardiacG2P — curated gene–disease relationships.
* AlphaMissense — general missense pathogenicity predictions.

Raw datasets are not stored directly in this repository. Download scripts, source identifiers, provenance information, and preprocessing instructions will be provided to support reproducibility.

## Repository structure

```text
data/           Data directories and provenance documentation
notebooks/      Exploratory analysis and modeling notebooks
src/cardiovus/  Reusable Python package
outputs/        Figures, predictions, embeddings, and trained models
app/            Interactive demonstration
reports/        Project reports and supporting documentation
tests/          Automated tests
```

## Planned notebooks

1. `01_eda_functional.ipynb`
   Data acquisition, quality control, functional EDA, and biological interpretation.

2. `02_features_modeling.ipynb`
   Feature engineering, ESM-2 embeddings, cross-validation, XGBoost modeling, interpretability, and external validation.

## Reproducibility

The project uses a fixed random seed and records all data accessions, preprocessing decisions, package versions, model parameters, and validation splits.

## Current status

Repository initialization and data-source selection.

## License

The source code is released under the MIT License unless otherwise stated. External datasets retain their original licenses and terms of use.
