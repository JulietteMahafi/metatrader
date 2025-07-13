import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import logging
import math
import os
from collections import OrderedDict

# --- Configuration ---
# Log file will be created in the same directory as the script
logging.basicConfig(level=logging.INFO, filename='model_server.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the shared directory for communication
# Get the AppData directory and build the correct, absolute path
appdata_dir = os.getenv('APPDATA')
if not appdata_dir:
    logging.error("FATAL: APPDATA environment variable not found.")
    raise ValueError("APPDATA environment variable not found.")
    
COMM_DIR = Path(appdata_dir) / "MetaQuotes/Terminal/Common/MQL5/Files"

# Ensure the directory exists
COMM_DIR.mkdir(parents=True, exist_ok=True)

# Define file paths for communication
INPUT_FILE = COMM_DIR / "mt5_request.json"
OUTPUT_FILE = COMM_DIR / "mt5_response.json"
STOP_FILE = COMM_DIR / "stop_server.flag"

# --- Model Definition ---
# (This is the same architecture as before)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0: pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_heads: int, num_layers: int, d_ff: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, 'gelu', batch_first=True)
        final_norm_layer = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=final_norm_layer)
        self.output_layer = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_layer(x)
        return x

# --- Main Server Logic ---
def main():
    logging.info("Starting model server...")
    
    try:
        # The model artifacts should be in the COMM_DIR
        model_artifacts_dir = COMM_DIR
        
        # Load model artifacts
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        logging.info(f"Loading model artifacts from: {model_artifacts_dir.resolve()}")

        with open(model_artifacts_dir / "model_metadata.json", 'r') as f:
            model_metadata = json.load(f)
        
        num_features = model_metadata['num_features']
        num_classes = model_metadata['num_classes']

        # Set weights_only=True for security
        checkpoint = torch.load(model_artifacts_dir / "model_best.pt", map_location=device, weights_only=True)
        args_dict = checkpoint['args']
        
        model = SimpleTransformer(
            input_dim=num_features, d_model=args_dict['d_model'], n_heads=args_dict['n_heads'],
            num_layers=args_dict['num_layers'], d_ff=args_dict['d_ff'],
            num_classes=num_classes, dropout=args_dict['dropout']
        ).to(device)

        original_state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict((k.replace('_orig_mod.', ''), v) for k, v in original_state_dict.items())
        model.load_state_dict(new_state_dict)
        model.eval()
        logging.info("Model loaded successfully.")

        scaler = joblib.load(model_artifacts_dir / "scaler.joblib")
        logging.info("Scaler loaded successfully.")
        
        label_encoder = joblib.load(model_artifacts_dir / "label_encoder.joblib")
        logging.info(f"Label encoder loaded. Classes: {list(label_encoder.classes_)}")

    except Exception as e:
        logging.error(f"FATAL: Could not load model artifacts. Server shutting down. Error: {e}", exc_info=True)
        return

    logging.info(f"--- Model server is running ---")
    logging.info(f"Watching for input file: {INPUT_FILE.resolve()}")

    while True:
        if STOP_FILE.exists():
            logging.info("Stop file detected. Shutting down server.")
            try:
                STOP_FILE.unlink()
            except OSError as e:
                logging.error(f"Error removing stop file: {e}")
            break

        if not INPUT_FILE.exists():
            time.sleep(0.05) # 50ms wait
            continue

        try:
            logging.info("Input file detected. Processing request.")
            # Add a small delay to ensure the file is fully written
            time.sleep(0.05)
            with open(INPUT_FILE, 'r') as f:
                data = json.load(f)
            
            # Preprocess
            df = pd.DataFrame(data['features'])
            numerical_features = model_metadata['numerical_features']
            df[numerical_features] = scaler.transform(df[numerical_features])
            input_tensor = torch.tensor(df.to_numpy()[np.newaxis, :, :], dtype=torch.float32).to(device)

            # Predict
            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Prepare response
            prediction_idx = np.argmax(probabilities)
            signal = label_encoder.inverse_transform([prediction_idx])[0]
            confidence = float(probabilities[prediction_idx])

            response = {
                "signal": signal,
                "confidence": confidence,
                "probabilities": {
                    label_encoder.classes_[i]: float(probabilities[i]) for i in range(len(probabilities))
                }
            }
            
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(response, f)
            
            logging.info(f"Prediction complete: {signal} ({confidence:.4f}). Response written.")

        except Exception as e:
            logging.error(f"Error processing request: {e}", exc_info=True)
        
        finally:
            # Clean up input file to signal completion
            try:
                INPUT_FILE.unlink()
            except OSError as e:
                logging.error(f"Error removing input file: {e}")


if __name__ == "__main__":
    main()
