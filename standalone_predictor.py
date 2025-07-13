#!/usr/bin/env python3
"""
Standalone Predictor for MT5 Integration
This script can be compiled to .exe and called from MT5 via ShellExecute()
"""

import sys
import os
import json
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import math
from collections import OrderedDict
import argparse

# Model definition (same as in FIXED_model_server.py)
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

def load_model(model_dir: Path):
    """Load the trained model and preprocessing artifacts"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model metadata
        with open(model_dir / "model_metadata.json", 'r') as f:
            model_metadata = json.load(f)
        
        num_features = model_metadata['num_features']
        num_classes = model_metadata['num_classes']

        # Load model checkpoint
        checkpoint = torch.load(model_dir / "model_best.pt", map_location=device, weights_only=True)
        args_dict = checkpoint['args']
        
        # Create and load model
        model = SimpleTransformer(
            input_dim=num_features, d_model=args_dict['d_model'], n_heads=args_dict['n_heads'],
            num_layers=args_dict['num_layers'], d_ff=args_dict['d_ff'],
            num_classes=num_classes, dropout=args_dict['dropout']
        ).to(device)

        # Load state dict
        original_state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict((k.replace('_orig_mod.', ''), v) for k, v in original_state_dict.items())
        model.load_state_dict(new_state_dict)
        model.eval()

        # Load preprocessing artifacts
        scaler = joblib.load(model_dir / "scaler.joblib")
        label_encoder = joblib.load(model_dir / "label_encoder.joblib")
        
        return model, scaler, label_encoder, model_metadata, device
    
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return None, None, None, None, None

def predict(features_data, model, scaler, label_encoder, model_metadata, device):
    """Make prediction on input features"""
    try:
        # Preprocess
        df = pd.DataFrame(features_data)
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

        return {
            "signal": signal,
            "confidence": confidence,
            "probabilities": {
                label_encoder.classes_[i]: float(probabilities[i]) for i in range(len(probabilities))
            }
        }
    
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Standalone MT5 Predictor')
    parser.add_argument('--input', type=str, help='Input JSON file path')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--model-dir', type=str, help='Model directory path')
    
    args = parser.parse_args()
    
    # If no arguments provided, use default MT5 paths
    if not args.input:
        appdata_dir = os.getenv('APPDATA')
        if not appdata_dir:
            print("ERROR: APPDATA environment variable not found")
            sys.exit(1)
        
        comm_dir = Path(appdata_dir) / "MetaQuotes/Terminal/Common/MQL5/Files"
        args.input = str(comm_dir / "mt5_request.json")
        args.output = str(comm_dir / "mt5_response.json")
        args.model_dir = str(comm_dir)
    
    # Load model
    model, scaler, label_encoder, model_metadata, device = load_model(Path(args.model_dir))
    if model is None:
        sys.exit(1)
    
    # Read input
    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to read input file: {e}")
        sys.exit(1)
    
    # Make prediction
    result = predict(data['features'], model, scaler, label_encoder, model_metadata, device)
    
    # Write output
    try:
        with open(args.output, 'w') as f:
            json.dump(result, f)
        print(f"SUCCESS: Prediction written to {args.output}")
    except Exception as e:
        print(f"ERROR: Failed to write output: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()