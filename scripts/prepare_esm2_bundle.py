"""Download and save an offline ESM-2 bundle for Kaggle.

Run on a machine with internet access:

    python prepare_esm2_bundle.py

Output:
    kaggle_upload/esm2_t6_8M_UR50D/

Upload the generated folder as a separate Kaggle Dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

OUTPUT_DIR = (
    Path("kaggle_upload")
    / "esm2_t6_8M_UR50D"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print(f"Downloading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

model.save_pretrained(
    OUTPUT_DIR,
    safe_serialization=True,
)

metadata = {
    "model_name": MODEL_NAME,
    "transformers_version": (
        transformers.__version__
    ),
    "torch_version": torch.__version__,
    "hidden_size": int(
        model.config.hidden_size
    ),
    "max_position_embeddings": int(
        model.config.max_position_embeddings
    ),
}

with (
    OUTPUT_DIR / "bundle_metadata.json"
).open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        indent=2,
    )

print(
    f"Offline bundle saved to: "
    f"{OUTPUT_DIR.resolve()}"
)
print("Files:")

for file_path in sorted(
    OUTPUT_DIR.iterdir()
):
    print(
        f"- {file_path.name}: "
        f"{file_path.stat().st_size:,} bytes"
    )
