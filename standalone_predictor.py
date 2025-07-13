#!/usr/bin/env python3
"""
Standalone Predictor for JulietteV2 Model
This script can be compiled to an executable using PyInstaller
Usage: standalone_predictor.exe <input_json_path> <output_json_path>
"""

import sys
import json
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import math
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# Model Architecture (must match trained model exactly)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0: 
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_heads: int, num_layers: int, 
                 d_ff: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, 'gelu', batch_first=True
        )
        final_norm_layer = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, norm=final_norm_layer
        )
        self.output_layer = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_layer(x)
        return x

def main():
    # Parse command line arguments
    if len(sys.argv) != 3:
        print("Usage: standalone_predictor.exe <input_json_path> <output_json_path>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    try:
        # Load model artifacts from bundled resources
        # When compiled with PyInstaller, use _MEIPASS
        if hasattr(sys, '_MEIPASS'):
            base_path = Path(sys._MEIPASS)
        else:
            # For development, assume files are in current directory
            base_path = Path(__file__).parent
        
        # Load model configuration
        with open(base_path / "model_metadata.json", 'r') as f:
            model_metadata = json.load(f)
        
        num_features = model_metadata['num_features']
        num_classes = model_metadata['num_classes']
        
        # Load model
        device = torch.device("cpu")  # Use CPU for compatibility
        checkpoint = torch.load(
            base_path / "model_best.pt", 
            map_location=device, 
            weights_only=True
        )
        args_dict = checkpoint['args']
        
        model = SimpleTransformer(
            input_dim=num_features,
            d_model=args_dict['d_model'],
            n_heads=args_dict['n_heads'],
            num_layers=args_dict['num_layers'],
            d_ff=args_dict['d_ff'],
            num_classes=num_classes,
            dropout=args_dict['dropout']
        ).to(device)
        
        # Fix state dict keys
        original_state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict(
            (k.replace('_orig_mod.', ''), v) for k, v in original_state_dict.items()
        )
        model.load_state_dict(new_state_dict)
        model.eval()
        
        # Load preprocessing tools
        scaler = joblib.load(base_path / "scaler.joblib")
        label_encoder = joblib.load(base_path / "label_encoder.joblib")
        
        # Read input data
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Preprocess
        df = pd.DataFrame(data['features'])
        numerical_features = model_metadata['numerical_features']
        df[numerical_features] = scaler.transform(df[numerical_features])
        input_tensor = torch.tensor(
            df.to_numpy()[np.newaxis, :, :], 
            dtype=torch.float32
        ).to(device)
        
        # Make prediction
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
                label_encoder.classes_[i]: float(probabilities[i]) 
                for i in range(len(probabilities))
            }
        }
        
        # Write output
        with open(output_path, 'w') as f:
            json.dump(response, f)
        
        # Return 0 for success
        sys.exit(0)
        
    except Exception as e:
        # Write error to output file
        error_response = {
            "error": str(e),
            "signal": "HOLD",
            "confidence": 0.0
        }
        with open(output_path, 'w') as f:
            json.dump(error_response, f)
        sys.exit(1)

if __name__ == "__main__":
    main()