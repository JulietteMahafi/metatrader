#!/usr/bin/env python3
"""
Standalone JulietteV2 Predictor for MT5 Integration
Designed to be compiled to .exe using PyInstaller
"""

import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import json
import sys
import os
from pathlib import Path
import argparse
import math
from collections import OrderedDict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class JulietteV2Predictor:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.model_metadata = None
        self.load_model()
    
    def load_model(self):
        """Load model artifacts"""
        try:
            # Load metadata
            with open(self.model_dir / "model_metadata.json", 'r') as f:
                self.model_metadata = json.load(f)
            
            # Load model
            checkpoint = torch.load(
                self.model_dir / "model_best.pt", 
                map_location=self.device,
                weights_only=True
            )
            
            args_dict = checkpoint['args']
            self.model = SimpleTransformer(
                input_dim=self.model_metadata['num_features'],
                d_model=args_dict['d_model'],
                n_heads=args_dict['n_heads'],
                num_layers=args_dict['num_layers'],
                d_ff=args_dict['d_ff'],
                num_classes=self.model_metadata['num_classes'],
                dropout=args_dict['dropout']
            ).to(self.device)
            
            # Handle state dict (remove _orig_mod. prefix if present)
            original_state_dict = checkpoint['model_state_dict']
            new_state_dict = OrderedDict()
            for k, v in original_state_dict.items():
                key = k.replace('_orig_mod.', '')
                new_state_dict[key] = v
            
            self.model.load_state_dict(new_state_dict)
            self.model.eval()
            
            # Load preprocessing artifacts
            self.scaler = joblib.load(self.model_dir / "scaler.joblib")
            self.label_encoder = joblib.load(self.model_dir / "label_encoder.joblib")
            
            logging.info(f"Model loaded successfully on {self.device}")
            logging.info(f"Classes: {list(self.label_encoder.classes_)}")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, features_json: str) -> dict:
        """Make prediction from JSON features"""
        try:
            # Parse input
            data = json.loads(features_json)
            df = pd.DataFrame(data['features'])
            
            # Preprocess
            numerical_features = self.model_metadata['numerical_features']
            df[numerical_features] = self.scaler.transform(df[numerical_features])
            
            # Convert to tensor
            input_tensor = torch.tensor(
                df.to_numpy()[np.newaxis, :, :], 
                dtype=torch.float32
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Prepare response
            prediction_idx = np.argmax(probabilities)
            signal = self.label_encoder.inverse_transform([prediction_idx])[0]
            confidence = float(probabilities[prediction_idx])
            
            return {
                "signal": signal,
                "confidence": confidence,
                "probabilities": {
                    self.label_encoder.classes_[i]: float(probabilities[i]) 
                    for i in range(len(probabilities))
                }
            }
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='JulietteV2 Standalone Predictor')
    parser.add_argument('--model-dir', required=True, help='Path to model artifacts directory')
    parser.add_argument('--input', help='Input JSON file path')
    parser.add_argument('--json', help='Input JSON string')
    parser.add_argument('--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = JulietteV2Predictor(args.model_dir)
    
    # Get input
    if args.input:
        with open(args.input, 'r') as f:
            input_json = f.read()
    elif args.json:
        input_json = args.json
    else:
        # Read from stdin
        input_json = sys.stdin.read()
    
    # Make prediction
    result = predictor.predict(input_json)
    
    # Output result
    result_json = json.dumps(result, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result_json)
    else:
        print(result_json)

if __name__ == "__main__":
    main()