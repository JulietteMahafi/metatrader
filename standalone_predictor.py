import argparse
import json
import sys
import os
import logging
from pathlib import Path
from collections import OrderedDict

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import joblib  # type: ignore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Model definition (identical to SimpleTransformer in FIXED_model_server.py)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers, nn.LayerNorm(d_model))
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.output_layer(x)


# ---------------------------------------------------------------------------
# Helper – resolve COMM_DIR similar to FIXED_model_server but one-off
# ---------------------------------------------------------------------------
if os.name == "nt" and os.getenv("APPDATA"):
    COMM_DIR = Path(os.getenv("APPDATA")) / "MetaQuotes/Terminal/Common/MQL5/Files"  # type: ignore[arg-type]
else:
    COMM_DIR = Path(__file__).resolve().parent / "comm_dir"
COMM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    with open(COMM_DIR / "model_metadata.json", "r") as f:
        model_metadata = json.load(f)

    num_features = model_metadata["num_features"]
    num_classes = model_metadata["num_classes"]

    checkpoint = torch.load(COMM_DIR / "model_best.pt", map_location=device, weights_only=True)
    args_dict = checkpoint["args"]

    MODEL = SimpleTransformer(
        input_dim=num_features,
        d_model=args_dict["d_model"],
        n_heads=args_dict["n_heads"],
        num_layers=args_dict["num_layers"],
        d_ff=args_dict["d_ff"],
        num_classes=num_classes,
        dropout=args_dict.get("dropout", 0.1),
    ).to(device)

    new_state_dict = OrderedDict(
        (k.replace("_orig_mod.", ""), v) for k, v in checkpoint["model_state_dict"].items()
    )
    MODEL.load_state_dict(new_state_dict, strict=True)
    MODEL.eval()

    SCALER = joblib.load(COMM_DIR / "scaler.joblib")
    LABEL_ENCODER = joblib.load(COMM_DIR / "label_encoder.joblib")

    logging.info("Artifacts loaded successfully from %s", COMM_DIR)
except Exception as err:  # pragma: no cover
    logging.error("Failed to load model artifacts: %s", err, exc_info=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict(request_json: dict) -> dict:
    """Perform a single prediction and return dictionary with signal & probs."""
    features = request_json["features"]  # Expect list[dict[str, float]]
    df = pd.DataFrame(features)
    numerical_features = model_metadata["numerical_features"]
    df[numerical_features] = SCALER.transform(df[numerical_features])

    tensor_in = torch.tensor(df.to_numpy()[np.newaxis, :, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = MODEL(tensor_in)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    signal = LABEL_ENCODER.inverse_transform([pred_idx])[0]

    return {
        "signal": signal,
        "confidence": float(probs[pred_idx]),
        "probabilities": {LABEL_ENCODER.classes_[i]: float(probs[i]) for i in range(len(probs))},
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="JulietteV2 standalone predictor (one-shot)")
    parser.add_argument("--input", "-i", help="Path to JSON file; if omitted read stdin")
    parser.add_argument("--output", "-o", help="Path to write JSON; if omitted write to stdout")

    args = parser.parse_args()

    # Read request JSON
    try:
        if args.input:
            with open(args.input, "r", encoding="utf-8") as f_in:
                request = json.load(f_in)
        else:
            request = json.load(sys.stdin)
    except Exception as read_err:
        logging.error("Failed to read input JSON: %s", read_err)
        sys.exit(2)

    # Predict
    try:
        response = predict(request)
    except Exception as pred_err:
        logging.error("Prediction failed: %s", pred_err, exc_info=True)
        sys.exit(3)

    # Write output
    try:
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f_out:
                json.dump(response, f_out)
        else:
            json.dump(response, sys.stdout)
    except Exception as write_err:
        logging.error("Failed to write response JSON: %s", write_err)
        sys.exit(4)


if __name__ == "__main__":
    main()