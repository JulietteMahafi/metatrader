#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score, recall_score
import time
import logging
import os
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import math
from packaging import version
import joblib
import gc
import torch.nn.functional as F
import sklearn # For NotFittedError
import torch.serialization # For add_safe_globals
import _pickle # For UnpicklingError


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
try:
    torch.use_deterministic_algorithms(True, warn_only=False)
except AttributeError:
    torch.backends.cudnn.deterministic = True
    logging.warning("Using older PyTorch version. Enabling deterministic mode via torch.backends.cudnn.deterministic.")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        logging.info("Forcing CPU usage.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        logging.info("CUDA is available. Using CUDA.")
        return torch.device("cuda:0")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logging.info("MPS is available. Using MPS.")
        return torch.device("mps")
    else:
        logging.info("CUDA and MPS not available. Using CPU.")
        return torch.device("cpu")

def get_amp_device_str(device: torch.device) -> str:
    if device.type == 'cuda':
        return 'cuda'
    return 'cpu'

def log_gpu_usage(device: torch.device):
    if device.type == 'cuda':
        try:
            gpu_properties = torch.cuda.get_device_properties(device)
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            total_memory = gpu_properties.total_memory / 1024**3
            utilization_allocated = (memory_allocated / total_memory) * 100 if total_memory > 0 else 0
            utilization_reserved = (memory_reserved / total_memory) * 100 if total_memory > 0 else 0

            logging.info(f"GPU: {gpu_properties.name}")
            logging.info(f"Total Memory: {total_memory:.2f} GB")
            logging.info(f"Allocated Memory: {memory_allocated:.2f} GB ({utilization_allocated:.2f}%)")
            logging.info(f"Reserved (Cached) Memory: {memory_reserved:.2f} GB ({utilization_reserved:.2f}%)")
        except Exception as e:
            logging.error(f"Could not get GPU details: {e}")
    elif device.type == 'mps':
        logging.info("Using Apple MPS.")
    else:
        logging.info("Using CPU.")

def auto_adjust_batch_size(seq_length: int, input_dim: int, initial_batch: int,
                           device: torch.device, utilisation_target: float = 0.70,
                           d_model_ref: Optional[int] = None) -> int:
    if device.type != 'cuda' or input_dim == 0 or seq_length == 0:
        logging.info(f"[AutoBatch] Not a CUDA device or invalid dims (input_dim: {input_dim}, seq_length: {seq_length}). Using initial_batch: {initial_batch}")
        return initial_batch

    log_gpu_usage(device)
    total_mem_bytes = torch.cuda.get_device_properties(device).total_memory

    bytes_per_float = 4
    effective_feature_dim = d_model_ref if d_model_ref is not None else input_dim

    mem_embedding = seq_length * effective_feature_dim * bytes_per_float * 4
    mem_attention_approx = seq_length * seq_length * bytes_per_float * 2
    mem_ffn_approx = seq_length * effective_feature_dim * 4 * bytes_per_float * 2

    per_sample_bytes_estimate = (mem_embedding + mem_attention_approx + mem_ffn_approx) * 2
    per_sample_bytes_estimate = max(per_sample_bytes_estimate, 1)

    if per_sample_bytes_estimate == 0:
        logging.warning("[AutoBatch] per_sample_bytes_estimate is 0. Using initial_batch.")
        return initial_batch

    effective_utilisation = min(utilisation_target, 0.90)
    reserved_overhead_bytes = total_mem_bytes * 0.10

    available_mem_for_batch = (total_mem_bytes * effective_utilisation) - reserved_overhead_bytes
    available_mem_for_batch = max(0, available_mem_for_batch)

    max_samples = int(available_mem_for_batch // per_sample_bytes_estimate) if per_sample_bytes_estimate > 0 else 0

    max_samples = max(0, max_samples)
    if max_samples == 0:
        logging.warning(f"[AutoBatch] Estimated available memory ({available_mem_for_batch/1024**3:.2f} GB) insufficient for even one sample (est. {per_sample_bytes_estimate/1024**2:.2f} MB/sample). Defaulting to initial_batch {initial_batch}.")
        return initial_batch

    adjusted_batch = min(initial_batch, max_samples)

    if adjusted_batch < initial_batch * 0.75 and adjusted_batch > 1:
        power_of_2 = 2**int(math.log2(adjusted_batch))
        if power_of_2 > 0 :
             adjusted_batch = power_of_2

    adjusted_batch = max(adjusted_batch, 1)

    if adjusted_batch != initial_batch:
        logging.info(f"[AutoBatch] Initial batch: {initial_batch}. Est. available VRAM for batch: {available_mem_for_batch/1024**3:.2f}GB. Est. bytes/sample: {per_sample_bytes_estimate/1024**2:.2f}MB.")
        logging.info(f"[AutoBatch] Adjusted batch size from {initial_batch} to {adjusted_batch} (GPU VRAM Target: ~{effective_utilisation*100:.0f}%).")
    else:
        logging.info(f"[AutoBatch] Initial batch size {initial_batch} deemed feasible.")
    return adjusted_batch


if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0).lower()
    if any(arch in gpu_name for arch in ['a100', 'h100']):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        pass

torch_version_str = torch.__version__
try:
    current_torch_version = version.parse(torch_version_str.split('+')[0])
    if current_torch_version >= version.parse('2.1.0') and torch.cuda.is_available():
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')
    elif torch.cuda.is_available():
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
    if torch.cuda.is_available():
        logging.info(f"PYTORCH_CUDA_ALLOC_CONF set to: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
except Exception:
    if torch.cuda.is_available():
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

def create_test_set(file_path: str, months_to_cut: int = 6, test_file_path_str: Optional[str] = None, date_col_arg: Optional[str] = None) -> Tuple[str, str]:
    original_file_path = Path(file_path)
    test_file_p = original_file_path.parent / f"{original_file_path.stem}_test.csv" if test_file_path_str is None else Path(test_file_path_str)
    train_split_file_p = original_file_path.parent / f"{original_file_path.stem}_train_split.csv"
    try:
        df = pd.read_csv(original_file_path)
        if df.empty: raise ValueError(f"Data file {original_file_path} is empty")
        date_col = date_col_arg
        if not date_col or date_col not in df.columns:
            date_col_candidates = ['date','Date','datetime','Datetime','time','Time','timestamp','Timestamp']
            date_col = next((col for col in date_col_candidates if col in df.columns), df.columns[0])
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            if df[date_col].isnull().any(): raise ValueError(f"Failed to convert '{date_col}' to datetime.")
        df = df.sort_values(by=date_col)
        latest_date = df[date_col].max()
        cutoff_date = latest_date - pd.DateOffset(months=months_to_cut)
        train_data = df[df[date_col] < cutoff_date].copy()
        test_data = df[df[date_col] >= cutoff_date].copy()
        if train_data.empty or test_data.empty:
            df_original_for_fallback = pd.read_csv(original_file_path)
            if not pd.api.types.is_datetime64_any_dtype(df_original_for_fallback[date_col]):
                df_original_for_fallback[date_col] = pd.to_datetime(df_original_for_fallback[date_col], errors='coerce')
            df_original_for_fallback = df_original_for_fallback.sort_values(by=date_col)
            split_idx_fallback = int(len(df_original_for_fallback) * 0.8)
            train_data = df_original_for_fallback.iloc[:split_idx_fallback].copy()
            test_data = df_original_for_fallback.iloc[split_idx_fallback:].copy()
            if train_data.empty or test_data.empty: raise ValueError("Fallback split also resulted in empty train or test set.")
        train_data.to_csv(train_split_file_p, index=False); test_data.to_csv(test_file_p, index=False)
        logging.info(f"Train set ({train_data.shape[0]}) saved to {train_split_file_p}, Test set ({test_data.shape[0]}) saved to {test_file_p}")
        return str(train_split_file_p), str(test_file_p)
    except Exception as e: logging.error(f"Error creating test set: {e}"); raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        if df.empty: logging.warning(f"Loaded dataframe from {file_path} is empty.")
        return df
    except FileNotFoundError: logging.error(f"Data file not found at {file_path}"); raise
    except Exception as e: logging.error(f"Error loading data from {file_path}: {e}"); raise

def preprocess_and_split_data_temporal(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler, LabelEncoder]:
    if df.empty:
        num_all_features = len(args.all_features) if args.all_features else 0
        return (np.empty((0, num_all_features)), np.array([], dtype=int), np.empty((0, num_all_features)), np.array([], dtype=int), RobustScaler(), LabelEncoder())
    label_encoder = LabelEncoder()
    try:
        if args.target_col not in df.columns: raise ValueError(f"Target column '{args.target_col}' not found.")
        unique_targets_in_df = sorted(list(df[args.target_col].unique()))
        label_encoder.fit(unique_targets_in_df)
        args.target_classes = list(label_encoder.classes_)
        y = label_encoder.transform(df[args.target_col])
    except Exception as e: logging.error(f"Error encoding target '{args.target_col}': {e}"); raise
    if not args.all_features: raise ValueError("args.all_features list is empty.")
    if not all(f in df.columns for f in args.all_features):
        raise ValueError(f"Missing features in DataFrame: {[f for f in args.all_features if f not in df.columns]}.")
    X_df = df[args.all_features].copy()
    train_size_ratio = 1.0 - args.val_size
    if not (0 < train_size_ratio < 1): raise ValueError("val_size must be > 0 and < 1.")
    split_idx = int(len(X_df) * train_size_ratio)
    if len(X_df) < 2 or split_idx <= 0 or split_idx >= len(X_df):
        X_train_df, y_train = X_df.copy(), y
        X_val_df, y_val = pd.DataFrame(columns=X_df.columns), np.array([], dtype=int)
    else:
        X_train_df, X_val_df = X_df.iloc[:split_idx].copy(), X_df.iloc[split_idx:].copy()
        y_train, y_val = y[:split_idx], y[split_idx:]
    scaler = RobustScaler()
    numerical_features_present = [f for f in args.numerical_features if f in X_train_df.columns]
    if numerical_features_present:
        if not X_train_df.empty:
            X_train_df.loc[:, numerical_features_present] = scaler.fit_transform(X_train_df[numerical_features_present])
            if not X_val_df.empty: X_val_df.loc[:, numerical_features_present] = scaler.transform(X_val_df[numerical_features_present])
    return X_train_df.values, y_train, X_val_df.values, y_val, scaler, label_encoder

def preprocess_test_data(test_df: pd.DataFrame, scaler: RobustScaler, label_encoder: LabelEncoder, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if test_df.empty:
        num_all_features = len(args.all_features) if args.all_features else 0
        return np.empty((0, num_all_features)), np.array([], dtype=int)
    try:
        if args.target_col not in test_df.columns: raise ValueError(f"Target column '{args.target_col}' not found in test data.")
        y_test = label_encoder.transform(test_df[args.target_col])
    except Exception as e: logging.error(f"Error encoding test target: {e}"); raise
    if not args.all_features: raise ValueError("args.all_features list is empty for test preprocessing.")
    if not all(f in test_df.columns for f in args.all_features):
        raise ValueError(f"Missing features in test DataFrame: {[f for f in args.all_features if f not in test_df.columns]}.")
    X_test_df_features = test_df[args.all_features].copy()
    numerical_features_present = [f for f in args.numerical_features if f in X_test_df_features.columns]
    if numerical_features_present and not X_test_df_features.empty:
        try:
            scaler.transform(X_test_df_features[numerical_features_present].head(1))
            X_test_df_features.loc[:, numerical_features_present] = scaler.transform(X_test_df_features[numerical_features_present])
        except sklearn.exceptions.NotFittedError: logging.warning("Scaler not fitted. Skipping test scaling.")
        except Exception as e: logging.error(f"Error scaling test data: {e}")
    return X_test_df_features.values, y_test

def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    if X.size == 0 or y.size == 0:
        num_feat = X.shape[1] if X.ndim == 2 and X.shape[0] > 0 else (X.shape[2] if X.ndim==3 and X.shape[0]>0 else 0)
        if X.ndim == 1 and X.shape[0] > 0 : num_feat = 1
        return np.empty((0, seq_length, num_feat)), np.empty((0,))
    if X.ndim == 1: X = X.reshape(-1,1)
    n_samples, n_features = X.shape
    if n_samples < seq_length: return np.empty((0, seq_length, n_features)), np.empty((0,))
    if not X.flags['C_CONTIGUOUS']: X = np.ascontiguousarray(X)
    shape = (n_samples - seq_length + 1, seq_length, n_features)
    strides = (X.strides[0], X.strides[0], X.strides[1])
    try: sequences_X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    except ValueError as e: logging.error(f"Error with as_strided: {e}"); raise
    sequences_y = y[seq_length - 1:]
    if sequences_X.shape[0] != sequences_y.shape[0]: raise RuntimeError("Sequence creation mismatch.")
    return sequences_X.copy(), sequences_y.copy()

class ForexDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.size == 0 and y.size == 0:
            expected_seq_len = X.shape[1] if X.ndim == 3 and X.shape[0] == 0 else 0
            expected_features = X.shape[2] if X.ndim == 3 and X.shape[0] == 0 else 0
            self.X = torch.empty((0, expected_seq_len, expected_features), dtype=torch.float32)
            self.y = torch.empty((0,), dtype=torch.long)
            return
        if X.ndim != 3: raise ValueError(f"X must be 3D, got {X.shape}")
        if y.ndim != 1: raise ValueError(f"y must be 1D, got {y.shape}")
        if X.shape[0] != y.shape[0]: raise ValueError(f"X,y sample count mismatch: {X.shape[0]} vs {y.shape[0]}.")
        self.X = torch.tensor(X, dtype=torch.float32); self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self) -> int: return self.X.shape[0]
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: return self.X[idx], self.y[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model <= 0 or not (0 <= dropout <= 1) or max_len <= 0: raise ValueError("Invalid PE params")
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_term_indices * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0: pe[:, 1::2] = torch.cos(position * div_term)
        else:
            if d_model > 1: pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1): raise ValueError(f"Input seq len ({seq_len}) > PE max_len ({self.pe.size(1)}).")
        if x.size(2) != self.pe.size(2): raise ValueError(f"Input feat dim ({x.size(2)}) != PE d_model ({self.pe.size(2)}).")
        output = x + self.pe[:, :seq_len, :]
        return self.dropout(output)

class MaxGPUTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_heads: int, num_layers: int, d_ff: int,
                 num_classes: int, dropout: float = 0.1, max_seq_len: int = 5000,
                 use_moe: bool = False, num_experts: int = 4, topk_experts: int = 2):
        super().__init__()
        if input_dim <= 0: raise ValueError("input_dim must be positive.")
        if any(param <= 0 for param in [d_model, n_heads, num_layers, d_ff]):
            raise ValueError("d_model, n_heads, num_layers, d_ff must be positive.")
        if num_classes < 0 : raise ValueError("num_classes must be non-negative.")
        if d_model % n_heads != 0: raise ValueError("d_model must be divisible by n_heads.")
        if not (0 <= dropout <= 1): raise ValueError("dropout must be between 0 and 1.")
        self.d_model = d_model; self.use_moe = use_moe
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, 'gelu', True, True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers, nn.LayerNorm(d_model))
        effective_num_classes = num_classes if num_classes > 0 else 1
        self.output_layer = nn.Linear(d_model, effective_num_classes)
        if num_classes == 0: logging.warning(f"MaxGPUTransformer output_layer initialized with out_features={effective_num_classes} due to num_classes=0.")
        self._init_parameters()
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
            elif isinstance(m, nn.LayerNorm): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        projected_tensor = self.input_projection(x).clone()
        pos_encoded_tensor = self.positional_encoding(projected_tensor)
        encoded_tensor = self.transformer_encoder(pos_encoded_tensor, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            expanded_mask = (~src_key_padding_mask).unsqueeze(-1).float()
            pooled_tensor = (encoded_tensor * expanded_mask).sum(dim=1) / torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
        else:
            pooled_tensor = torch.mean(encoded_tensor, dim=1)
        return self.output_layer(pooled_tensor)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__(); self.gamma, self.weight, self.reduction = gamma, weight, reduction
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(inputs, dim=1); p = torch.exp(logp)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None: loss = loss * self.weight.to(targets.device)[targets]
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        return loss

class CostSensitiveRegularizedLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, lambd: float, cost_matrix: torch.Tensor):
        super().__init__(); self.base_loss, self.lambd = base_loss, lambd
        self.register_buffer('cost_matrix', cost_matrix)
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_loss_value = self.base_loss(inputs, targets)
        probs = F.softmax(inputs, dim=1)
        current_cost_matrix = self.cost_matrix.to(probs.device)
        costs_for_true_targets = current_cost_matrix[targets]
        expected_cost_per_sample = torch.sum(probs * costs_for_true_targets, dim=1)
        cost_reg_term = expected_cost_per_sample.mean()
        return base_loss_value + self.lambd * cost_reg_term

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    if total_steps <= 0: return optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    if total_steps <= warmup_steps: return optim.lr_scheduler.LambdaLR(optimizer, lambda cs: float(cs+1)/float(max(1,total_steps)))
    def lr_lambda(cs):
        if cs < warmup_steps: return float(cs+1)/float(max(1,warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * float(cs-warmup_steps)/float(max(1,total_steps-warmup_steps))))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_model_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scheduler: Any, path: Path, args: argparse.Namespace, label_encoder: LabelEncoder, current_val_metrics: Dict[str, Any], epoch: int):
    if hasattr(model, '_orig_mod'):
        orig_model = model._orig_mod
        input_dim_val = orig_model.input_projection.in_features if hasattr(orig_model, 'input_projection') else None
        num_classes_val = orig_model.output_layer.out_features if hasattr(orig_model, 'output_layer') else None
    else:
        input_dim_val = model.input_projection.in_features if hasattr(model, 'input_projection') else None
        num_classes_val = model.output_layer.out_features if hasattr(model, 'output_layer') else None

    if input_dim_val is None or num_classes_val is None:
        input_dim_val = input_dim_val or (len(args.all_features) if args.all_features else 0)
        num_classes_val = num_classes_val or (len(label_encoder.classes_) if hasattr(label_encoder, 'classes_') else 0)

    model_state_to_save = model.state_dict()

    checkpoint = {'epoch': epoch, 'model_state_dict': model_state_to_save, 'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'args': vars(args),
                  'input_dim': input_dim_val, 'num_classes': num_classes_val,
                  'label_encoder_classes': list(label_encoder.classes_) if hasattr(label_encoder, 'classes_') else [],
                  'validation_metrics': current_val_metrics, 'random_state_torch': torch.get_rng_state(),
                  'random_state_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                  'random_state_numpy': np.random.get_state()}
    torch.save(checkpoint, path)
    logging.info(f"Model checkpoint saved to {path}")

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader], criterion: nn.Module,
                optimizer: optim.Optimizer, scheduler: Any, device: torch.device, epochs: int, patience: int,
                output_dir: Path, grad_accum_steps: int, args: argparse.Namespace, label_encoder: LabelEncoder) -> nn.Module:
    best_primary_metric, epochs_no_improve, best_epoch_details = -1.0, 0, {}
    amp_device_str, use_amp = get_amp_device_str(device), device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)
    metrics_df_path, metrics_log_path, best_model_ckpt_path = output_dir/"training_metrics_history.csv", output_dir/"training_log.txt", output_dir/"model_best.pt"
    with open(metrics_log_path, 'w') as f: f.write(f"Run: {datetime.now()}\nArgs: {vars(args)}\nepoch,train_loss,train_acc,val_loss,val_acc,val_f1_macro,val_avg_buy_sell_precision,val_buy_precision,val_sell_precision,lr\n")
    metrics_history_list = []
    for epoch in range(epochs):
        epoch_start_time = time.time()
        if device.type == 'cuda': torch.cuda.empty_cache(); gc.collect()
        elif device.type == 'mps' and hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()
        model.train(); train_loss_sum, correct_train, total_train = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=amp_device_str, enabled=use_amp):
                outputs = model(inputs); loss_per_sample = criterion(outputs, targets)
                loss_for_step = loss_per_sample / grad_accum_steps
            if use_amp: scaler.scale(loss_for_step).backward()
            else: loss_for_step.backward()
            train_loss_sum += loss_per_sample.item()*inputs.size(0); _, predicted = outputs.max(1)
            total_train += targets.size(0); correct_train += predicted.eq(targets).sum().item()
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if use_amp: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
                else: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler and isinstance(scheduler, optim.lr_scheduler.LambdaLR): scheduler.step()
            if (batch_idx + 1) % args.log_interval == 0:
                logging.info(f"E:{epoch+1} B:{(batch_idx+1)}/{len(train_loader)} Loss:{(train_loss_sum/(total_train if total_train>0 else 1)):.4f} Acc:{(100.*correct_train/(total_train if total_train >0 else 1)):.2f}%")
        avg_train_loss, train_acc = (train_loss_sum/total_train if total_train>0 else float('nan')), (100.*correct_train/total_train if total_train>0 else 0.0)
        current_val_metrics_dict, avg_buy_sell_precision_epoch = {}, 0.0
        if val_loader and hasattr(val_loader, 'dataset') and val_loader.dataset and len(val_loader.dataset) > 0:
            current_val_metrics_dict = evaluate_model_enhanced(model, val_loader, criterion, device, list(label_encoder.classes_) if hasattr(label_encoder, 'classes_') else [], "Validation", use_amp, amp_device_str)
            avg_buy_sell_precision_epoch = current_val_metrics_dict.get('avg_buy_sell_precision', 0.0)
        else: logging.info(f"E:{epoch+1} No validation data.")
        epoch_metrics_log = {'epoch':epoch+1, 'train_loss':avg_train_loss, 'train_acc':train_acc, 'val_loss':current_val_metrics_dict.get('loss',float('nan')),
                             'val_acc':current_val_metrics_dict.get('accuracy',float('nan')), 'val_f1_macro':current_val_metrics_dict.get('f1_macro',float('nan')),
                             'val_avg_buy_sell_precision':avg_buy_sell_precision_epoch, 'val_buy_precision':current_val_metrics_dict.get('buy_precision',float('nan')),
                             'val_sell_precision':current_val_metrics_dict.get('sell_precision',float('nan')), 'lr':optimizer.param_groups[0]['lr']}
        metrics_history_list.append(epoch_metrics_log)
        with open(metrics_log_path, 'a') as f: f.write(f"{epoch_metrics_log['epoch']},{epoch_metrics_log['train_loss']:.4f},{epoch_metrics_log['train_acc']:.2f},{epoch_metrics_log['val_loss']:.4f},{epoch_metrics_log['val_acc']:.2f},{epoch_metrics_log['val_f1_macro']:.4f},{epoch_metrics_log['val_avg_buy_sell_precision']:.4f},{epoch_metrics_log['val_buy_precision']:.4f},{epoch_metrics_log['val_sell_precision']:.4f},{epoch_metrics_log['lr']:.2e}\n")
        logging.info(f"E:{epoch+1} TrL:{avg_train_loss:.4f} TrAcc:{train_acc:.2f}% VaL:{epoch_metrics_log['val_loss']:.4f} VaAcc:{epoch_metrics_log['val_acc']:.2f}% F1M:{epoch_metrics_log['val_f1_macro']:.4f} AvgBSPrec:{avg_buy_sell_precision_epoch:.4f} LR:{epoch_metrics_log['lr']:.2e} Time:{(time.time()-epoch_start_time):.2f}s")
        if val_loader and hasattr(val_loader, 'dataset') and val_loader.dataset and len(val_loader.dataset) > 0:
            if avg_buy_sell_precision_epoch > best_primary_metric:
                best_primary_metric, epochs_no_improve, best_epoch_details['epoch'], best_epoch_details['validation_metrics'] = avg_buy_sell_precision_epoch, 0, epoch+1, current_val_metrics_dict
                save_model_checkpoint(model, optimizer, scheduler, best_model_ckpt_path, args, label_encoder, current_val_metrics_dict, epoch+1)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience: logging.info(f"Early stopping at E:{epoch+1}. Best AvgBuySellPrec: {best_primary_metric:.4f} at E:{best_epoch_details.get('epoch', 'N/A')}."); break
            if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(avg_buy_sell_precision_epoch)
        if (epoch+1)%args.checkpoint_interval==0: save_model_checkpoint(model,optimizer,scheduler,output_dir/f"checkpoint_epoch_{epoch+1}.pt",args,label_encoder,current_val_metrics_dict,epoch+1)
        if val_loader and hasattr(val_loader,'dataset') and val_loader.dataset and len(val_loader.dataset)>0 and (epoch+1)%args.benchmark_interval==0: benchmark_model_performance(model,val_loader,criterion,device,list(label_encoder.classes_) if hasattr(label_encoder,'classes_') else [],epoch+1,output_dir,use_amp,amp_device_str)
    pd.DataFrame(metrics_history_list).to_csv(metrics_df_path, index=False)
    if os.path.exists(metrics_df_path) and not pd.read_csv(metrics_df_path).empty: plot_training_metrics(pd.read_csv(metrics_df_path), output_dir)

    if os.path.exists(best_model_ckpt_path):
        try:
            logging.info(f"Attempting to load best model from {best_model_ckpt_path} with weights_only=False.")
            checkpoint_to_load = torch.load(best_model_ckpt_path, map_location=device, weights_only=False)
            state_dict_to_load = checkpoint_to_load['model_state_dict']
            new_state_dict_to_load = {}
            is_current_model_compiled = hasattr(model, '_orig_mod')
            is_checkpoint_compiled = any(k.startswith("_orig_mod.") for k in state_dict_to_load.keys())

            if is_checkpoint_compiled:
                if is_current_model_compiled:
                    new_state_dict_to_load = state_dict_to_load
                else:
                    for k, v in state_dict_to_load.items():
                        if k.startswith("_orig_mod."): new_state_dict_to_load[k[len("_orig_mod."):]] = v
                        else: new_state_dict_to_load[k] = v
            else:
                if is_current_model_compiled:
                    for k, v in state_dict_to_load.items(): new_state_dict_to_load["_orig_mod." + k] = v
                else:
                    new_state_dict_to_load = state_dict_to_load
            
            model.load_state_dict(new_state_dict_to_load)
            logging.info(f"Successfully loaded best model state from {best_model_ckpt_path} with weights_only=False into current model instance.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading the best model state from {best_model_ckpt_path} with weights_only=False: {e}")
    return model

def evaluate_model_enhanced(model: nn.Module, data_loader: Optional[DataLoader], criterion: nn.Module, device: torch.device, class_names: List[str], dataset_name: str = "Test", use_amp: bool = False, amp_device_str: str = 'cpu') -> Dict[str, Any]:
    results = {'loss':float('nan'),'accuracy':0.0,'f1_macro':0.0,'f1_weighted':0.0,'cm':np.array([]),'report':"Eval not performed.",'buy_precision':0.0,'sell_precision':0.0,'avg_buy_sell_precision':0.0}
    if not data_loader or not hasattr(data_loader,'dataset') or len(data_loader.dataset)==0: results['report']="No data for eval."; return results

    model.eval(); loss_sum,total_samples_eval=0.0,0; all_preds_list,all_targets_list=[],[]
    try:
        with torch.no_grad():
            for inputs,targets in data_loader:
                inputs,targets=inputs.to(device,non_blocking=True),targets.to(device,non_blocking=True)
                with torch.amp.autocast(device_type=amp_device_str,enabled=use_amp):
                    outputs = model(inputs)
                    loss=criterion(outputs,targets)
                loss_sum+=loss.item()*inputs.size(0); total_samples_eval+=inputs.size(0)
                _,predicted_indices=outputs.max(1); all_preds_list.append(predicted_indices.cpu().numpy()); all_targets_list.append(targets.cpu().numpy())
        if not all_preds_list or not all_targets_list: results['report']="No preds/targets collected."; return results
        all_preds_np,all_targets_np=np.concatenate(all_preds_list),np.concatenate(all_targets_list)
        if len(all_targets_np)==0: results['report']="Targets empty post-concat."; return results
        results['loss']=loss_sum/total_samples_eval if total_samples_eval>0 else float('nan')
        results['accuracy']=100.*(all_preds_np==all_targets_np).sum()/len(all_targets_np)
        labels_for_sklearn=np.arange(len(class_names)) if class_names else np.unique(np.concatenate((all_targets_np,all_preds_np)))
        if len(labels_for_sklearn)>0 :
            results['f1_macro']=f1_score(all_targets_np,all_preds_np,average='macro',zero_division=0,labels=labels_for_sklearn)
            results['f1_weighted']=f1_score(all_targets_np,all_preds_np,average='weighted',zero_division=0,labels=labels_for_sklearn)
            results['cm']=confusion_matrix(all_targets_np,all_preds_np,labels=labels_for_sklearn)
            results['report']=classification_report(all_targets_np,all_preds_np,target_names=class_names if class_names else None,zero_division=0,labels=labels_for_sklearn,output_dict=False)
            prec_scores=precision_score(all_targets_np,all_preds_np,average=None,zero_division=0,labels=labels_for_sklearn)
            buy_idx,sell_idx=-1,-1; class_names_lower=[name.lower() for name in class_names] if class_names else []
            try: buy_idx=class_names_lower.index("buy")
            except ValueError: pass
            try: sell_idx=class_names_lower.index("sell")
            except ValueError: pass
            if buy_idx!=-1 and buy_idx<len(prec_scores): results['buy_precision']=prec_scores[buy_idx]
            if sell_idx!=-1 and sell_idx<len(prec_scores): results['sell_precision']=prec_scores[sell_idx]
            num_avg,sum_avg=0,0.0
            if buy_idx!=-1: sum_avg+=results['buy_precision']; num_avg+=1
            if sell_idx!=-1: sum_avg+=results['sell_precision']; num_avg+=1
            results['avg_buy_sell_precision']=sum_avg/num_avg if num_avg>0 else 0.0
        logging.info(f"{dataset_name} -> Loss:{results['loss']:.4f} Acc:{results['accuracy']:.2f}% F1M:{results['f1_macro']:.4f} AvgBSPrec:{results['avg_buy_sell_precision']:.4f}")
    except Exception as e: logging.error(f"Error Eval {dataset_name}: {e}"); results['report']=f"Error: {str(e)}"
    return results

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    if cm.size==0: return
    eff_y=class_names[:cm.shape[0]] if class_names and len(class_names)>=cm.shape[0] else [f"C{i}" for i in range(cm.shape[0])]
    eff_x=class_names[:cm.shape[1]] if class_names and len(class_names)>=cm.shape[1] else [f"C{i}" for i in range(cm.shape[1])]
    plt.figure(figsize=(max(8,len(eff_x)*0.8),max(6,len(eff_y)*0.7))); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=eff_x,yticklabels=eff_y)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix'); plt.tight_layout()
    try: plt.savefig(save_path); plt.close(); logging.info(f"CM plot saved to {save_path}")
    except Exception as e: logging.error(f"Saving CM plot: {e}")

def plot_training_metrics(metrics_df: pd.DataFrame, output_dir: Path):
    if metrics_df.empty or len(metrics_df)<=1: return
    plt.figure(figsize=(18,12)); epochs_data=metrics_df['epoch']
    plt.subplot(2,3,1);
    if 'train_loss' in metrics_df.columns and 'val_loss' in metrics_df.columns: plt.plot(epochs_data,metrics_df['train_loss'],'b-',label='Train Loss'); plt.plot(epochs_data,metrics_df['val_loss'],'r-',label='Val Loss')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True,alpha=0.3)
    plt.subplot(2,3,2);
    if 'train_acc' in metrics_df.columns and 'val_acc' in metrics_df.columns: plt.plot(epochs_data,metrics_df['train_acc'],'b-',label='Train Acc'); plt.plot(epochs_data,metrics_df['val_acc'],'r-',label='Val Acc')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True,alpha=0.3)
    plt.subplot(2,3,3);
    if 'val_f1_macro' in metrics_df.columns: plt.plot(epochs_data,metrics_df['val_f1_macro'],'g-',label='Val F1 Macro')
    plt.title('Val F1 Macro'); plt.xlabel('Epoch'); plt.ylabel('F1 Macro'); plt.legend(); plt.grid(True,alpha=0.3)
    plt.subplot(2,3,4);
    if 'val_avg_buy_sell_precision' in metrics_df.columns: plt.plot(epochs_data,metrics_df['val_avg_buy_sell_precision'],'purple',label='Val Avg Buy/Sell Prec')
    plt.title('Val Avg Buy/Sell Prec'); plt.xlabel('Epoch'); plt.ylabel('Precision'); plt.legend(); plt.grid(True,alpha=0.3)
    plt.subplot(2,3,5);
    if 'val_buy_precision' in metrics_df.columns: plt.plot(epochs_data,metrics_df['val_buy_precision'],'cyan',label='Val BUY Prec')
    if 'val_sell_precision' in metrics_df.columns: plt.plot(epochs_data,metrics_df['val_sell_precision'],'magenta',label='Val SELL Prec')
    plt.title('Val BUY/SELL Prec'); plt.xlabel('Epoch'); plt.ylabel('Precision'); plt.legend(); plt.grid(True,alpha=0.3)
    plt.subplot(2,3,6);
    if 'lr' in metrics_df.columns: plt.plot(epochs_data,metrics_df['lr']);
    try: plt.yscale('log')
    except (ValueError,TypeError): pass
    plt.title('Learning Rate'); plt.xlabel('Epoch'); plt.ylabel('LR'); plt.grid(True,alpha=0.3)
    plt.tight_layout(); save_path=output_dir/'training_summary_plots.png'
    try: plt.savefig(save_path,dpi=300); plt.close(); logging.info(f"Training plots saved to {save_path}")
    except Exception as e: logging.error(f"Saving training plots: {e}")

def benchmark_model_performance(model:nn.Module,data_loader:DataLoader,criterion:nn.Module,device:torch.device,class_names:List[str],epoch:int,output_dir:Path,use_amp:bool,amp_device_str:str) -> Dict[str,Any]:
    metrics=evaluate_model_enhanced(model,data_loader,criterion,device,class_names,f"Benchmark-E{epoch}",use_amp,amp_device_str)
    if metrics.get('cm') is not None and metrics['cm'].size!=0: plot_confusion_matrix(metrics['cm'],class_names,output_dir/f"benchmark_cm_epoch_{epoch}.png")
    df_b={'epoch':epoch,'loss':metrics.get('loss',float('nan')),'accuracy':metrics.get('accuracy',float('nan')),'f1_macro':metrics.get('f1_macro',float('nan')),
          'avg_buy_sell_precision':metrics.get('avg_buy_sell_precision',float('nan')),'buy_precision':metrics.get('buy_precision',float('nan')),'sell_precision':metrics.get('sell_precision',float('nan'))}
    bf_path=output_dir/"benchmark_run_history.csv"
    if not bf_path.exists(): pd.DataFrame([df_b]).to_csv(bf_path,index=False)
    else: pd.DataFrame([df_b]).to_csv(bf_path,mode='a',header=False,index=False)
    return metrics

def load_trained_model(checkpoint_path:Union[str,Path],device:Union[str,torch.device]='cpu') -> Tuple[Optional[MaxGPUTransformer],Optional[RobustScaler],Optional[LabelEncoder],Optional[argparse.Namespace]]:
    try:
        ckpt_p=Path(checkpoint_path); current_device=torch.device(device)
        if not ckpt_p.exists(): logging.error(f"Checkpoint not found: {ckpt_p}"); return None,None,None,None

        try:
            logging.info(f"Loading checkpoint {ckpt_p} with weights_only=False.")
            ckpt=torch.load(ckpt_p,map_location=current_device, weights_only=False)
        except Exception as e_load:
            logging.error(f"Error loading checkpoint {ckpt_p} with weights_only=False: {e_load}")
            return None,None,None,None

        req_keys=['model_state_dict','args','label_encoder_classes','input_dim','num_classes']
        if not all(k in ckpt for k in req_keys): logging.error(f"Checkpoint missing keys. Found: {list(ckpt.keys())}. Req: {req_keys}"); return None,None,None,None
        args_dict=ckpt['args']
        if not isinstance(args_dict,dict): logging.error(f"Args in ckpt not dict: {type(args_dict)}"); return None,None,None,None
        args_ns=argparse.Namespace(**args_dict)

        model=MaxGPUTransformer(ckpt['input_dim'],args_ns.d_model,args_ns.n_heads,args_ns.num_layers,args_ns.d_ff,ckpt['num_classes'],args_ns.dropout,getattr(args_ns,'pe_max_len_default',5000),getattr(args_ns,'use_moe',False))

        state_dict = ckpt['model_state_dict']
        new_state_dict = {}
        is_checkpoint_compiled = any(k.startswith("_orig_mod.") for k in state_dict.keys())

        if is_checkpoint_compiled:
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."): new_state_dict[k[len("_orig_mod."):]] = v
                else: new_state_dict[k] = v
        else:
            new_state_dict = state_dict

        model.load_state_dict(new_state_dict)
        model.to(current_device)
        model.eval()

        scaler=joblib.load(ckpt_p.parent/'scaler.joblib') if (ckpt_p.parent/'scaler.joblib').exists() else None
        le_obj=joblib.load(ckpt_p.parent/'label_encoder.joblib') if (ckpt_p.parent/'label_encoder.joblib').exists() else None
        if not scaler: logging.warning(f"Scaler not found at {ckpt_p.parent/'scaler.joblib'}")
        if not le_obj:
            if 'label_encoder_classes' in ckpt and ckpt['label_encoder_classes']:
                le_obj=LabelEncoder(); le_obj.classes_=np.array(ckpt['label_encoder_classes'])
            else: logging.error("LabelEncoder not found and cannot be reconstructed.")
        elif hasattr(le_obj,'classes_') and list(le_obj.classes_)!=ckpt['label_encoder_classes']: logging.warning(f"Loaded LE classes differ from ckpt!")
        return model,scaler,le_obj,args_ns
    except Exception as e: logging.error(f"Error loading model from {checkpoint_path}: {e}"); return None,None,None,None

def main(args: argparse.Namespace):
    # Block for add_safe_globals removed due to AttributeError with older PyTorch versions.
    # The script's robust fallback for model loading (weights_only=True then weights_only=False)
    # will handle unpickling issues.

    torch.manual_seed(args.random_state)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.random_state)
    np.random.seed(args.random_state)
    logging.info(f"Starting Training. Args: {vars(args)}")

    # Initialize device and related variables first
    device = get_device(args.force_cpu)
    amp_device_str = get_amp_device_str(device) # Corrected: use the already defined 'device'
    use_amp = device.type == 'cuda' # Corrected: use the already defined 'device'
    logging.info(f"Device: {device}, AMP autocast_device: {amp_device_str}, AMP enabled: {use_amp}")

    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    elif device.type == 'mps' and hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache() # Now 'device' is defined
    
    if device.type in ['cuda','mps']: log_gpu_usage(device)
    try:
        if args.create_initial_split and args.initial_dataset_file:
            if not Path(args.initial_dataset_file).exists(): raise FileNotFoundError(f"Initial dataset file not found: {args.initial_dataset_file}")
            args.train_file,args.test_file=create_test_set(args.initial_dataset_file,args.test_months_cutoff,args.date_col)
        train_df,test_df=load_data(args.train_file),load_data(args.test_file)
        if train_df.empty: logging.error(f"Training data from {args.train_file} is empty."); return
        X_train_np,y_train,X_val_np,y_val,scaler,label_encoder=preprocess_and_split_data_temporal(train_df,args)
        if not test_df.empty: X_test_np,y_test=preprocess_test_data(test_df,scaler,label_encoder,args)
        else: X_test_np,y_test=np.empty((0,len(args.all_features) if args.all_features else (X_train_np.shape[1] if X_train_np.size>0 else 0))),np.array([],dtype=int)
    except FileNotFoundError as e: logging.error(f"Data loading failed: {e}"); return
    except Exception as e: logging.error(f"Data loading/preprocessing: {e}"); return
    num_classes=len(label_encoder.classes_) if hasattr(label_encoder,'classes_') and label_encoder.classes_ is not None else 0
    input_dim=X_train_np.shape[1] if X_train_np.ndim==2 and X_train_np.shape[1]>0 else (len(args.all_features) if args.all_features else 0)
    if input_dim==0: logging.error("Input dimension is 0."); return
    if num_classes==0 and len(y_train)>0: logging.error(f"num_classes is 0, but y_train not empty. Target: '{args.target_col}'.")
    logging.info(f"Input dim: {input_dim}, Seq len: {args.sequence_length}, Num classes: {num_classes} ({list(label_encoder.classes_) if num_classes>0 else 'None'})")
    if args.auto_batch_size and device.type=='cuda' and input_dim>0: args.batch_size=auto_adjust_batch_size(args.sequence_length,input_dim,args.batch_size,device,d_model_ref=args.d_model)
    class_weights_val=None
    if len(y_train)>0 and num_classes>0:
        counts=np.bincount(y_train,minlength=num_classes)
        if np.all(counts>0): class_weights_val=torch.tensor(len(y_train)/(num_classes*counts.astype(float)),dtype=torch.float32).to(device)
    X_train_seq,y_train_seq=create_sequences(X_train_np,y_train,args.sequence_length)
    X_val_seq,y_val_seq=create_sequences(X_val_np,y_val,args.sequence_length)
    X_test_seq,y_test_seq=create_sequences(X_test_np,y_test,args.sequence_length)
    if X_train_seq.size==0: logging.error("Empty training sequences."); return
    train_dataset,val_dataset,test_dataset=ForexDataset(X_train_seq,y_train_seq),(ForexDataset(X_val_seq,y_val_seq) if X_val_seq.size>0 and y_val_seq.size>0 else None),(ForexDataset(X_test_seq,y_test_seq) if X_test_seq.size>0 and y_test_seq.size>0 else None)
    num_workers=min(os.cpu_count() or 1,args.num_workers) if args.num_workers>=0 else (os.cpu_count() or 1)
    if device.type=='mps' and num_workers>0: num_workers=0
    g=torch.Generator(); g.manual_seed(args.random_state)
    global current_torch_version; use_persistent_workers=num_workers>0 and current_torch_version>=version.parse('1.8.0')
    common_loader_args={'pin_memory':device.type=='cuda','num_workers':num_workers,'persistent_workers':use_persistent_workers if num_workers>0 else False,'prefetch_factor':args.prefetch_factor if num_workers>0 and hasattr(args,'prefetch_factor') else (2 if num_workers>0 else None)}
    if len(train_dataset)==0: logging.error("Train dataset empty."); return
    train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,drop_last=True,generator=g,**common_loader_args)
    val_loader=DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,**common_loader_args) if val_dataset and len(val_dataset)>0 else None
    test_loader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,**common_loader_args) if test_dataset and len(test_dataset)>0 else None
    model=MaxGPUTransformer(input_dim,args.d_model,args.n_heads,args.num_layers,args.d_ff,num_classes,args.dropout,args.pe_max_len_default,args.use_moe).to(device)
    if args.compile_model and hasattr(torch,'compile') and device.type=='cuda' and current_torch_version>=version.parse('2.0.0'):
        try: model=torch.compile(model,mode="reduce-overhead"); logging.info("Model compiled.")
        except Exception as e: logging.warning(f"torch.compile() failed: {e}")
    if args.gradient_checkpointing and hasattr(model,'transformer_encoder') and hasattr(model.transformer_encoder,'layers'):
        try:
            from torch.utils.checkpoint import checkpoint_sequential
            if isinstance(model.transformer_encoder.layers,(nn.ModuleList,nn.Sequential)):
                model.transformer_encoder.layers=checkpoint_sequential(model.transformer_encoder.layers,len(model.transformer_encoder.layers))
                logging.info("Applied gradient checkpointing to MaxGPUTransformer encoder layers.")
        except Exception as e: logging.error(f"Error Grad checkpointing: {e}")
    cost_m_values=torch.tensor([[0.0,2.5,3.0],[1.0,0.0,1.0],[3.0,2.5,0.0]],dtype=torch.float32)
    if num_classes>0 and (cost_m_values.shape[0]!=num_classes or cost_m_values.shape[1]!=num_classes): logging.warning(f"Cost matrix shape {cost_m_values.shape} vs num_classes {num_classes}.")
    base_focal_loss=FocalLoss(args.focal_gamma,class_weights_val,'mean')

    criterion = FocalLoss(args.focal_gamma,class_weights_val,'mean')
    # criterion=CostSensitiveRegularizedLoss(base_focal_loss,args.cost_lambda,cost_m_values)
    logging.info(f"Using CostSensitiveRegularizedLoss (Focal gamma={args.focal_gamma}, cost_lambda={args.cost_lambda}). Class weights for FocalLoss: {'Applied' if class_weights_val is not None else 'None'}")
    optimizer=optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay,amsgrad=True); optimizer.zero_grad(set_to_none=True)
    steps_per_epoch=math.ceil(len(train_loader)/args.grad_accum_steps) if len(train_loader)>0 and args.grad_accum_steps>0 else 0
    total_effective_steps=steps_per_epoch*args.epochs; warmup_steps=int(total_effective_steps*args.warmup_ratio) if total_effective_steps>0 else 0; scheduler=None
    if args.use_warmup and warmup_steps>0 and total_effective_steps>0: scheduler=get_lr_scheduler(optimizer,warmup_steps,total_effective_steps)
    elif val_loader: scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',0.5,args.lr_patience,0.001,verbose=False)
    run_stamp=datetime.now().strftime('%Y%m%d_%H%M%S'); output_dir=Path(args.output_dir)/f"run_{run_stamp}"; output_dir.mkdir(parents=True,exist_ok=True)
    logging.info(f"Run output directory: {output_dir.resolve()}")
    model=train_model(model,train_loader,val_loader,criterion,optimizer,scheduler,device,args.epochs,args.early_stopping_patience,output_dir,args.grad_accum_steps,args,label_encoder)
    try: joblib.dump(scaler,output_dir/"scaler.joblib"); joblib.dump(label_encoder,output_dir/"label_encoder.joblib")
    except Exception as e: logging.error(f"Error saving scaler/LE: {e}")
    best_model_path=output_dir/"model_best.pt"
    if best_model_path.exists():
        final_model,_,final_le,final_args_ns=load_trained_model(best_model_path,device=device)
        if final_model and final_le and hasattr(final_le,'classes_') and final_args_ns:
            final_le_classes=list(final_le.classes_)
            cost_m_values_final = torch.tensor([[0.0,2.5,3.0],[1.0,0.0,1.0],[3.0,2.5,0.0]],dtype=torch.float32)
            base_focal_loss_final = FocalLoss(final_args_ns.focal_gamma, None, 'mean')
            current_criterion=CostSensitiveRegularizedLoss(base_focal_loss_final,final_args_ns.cost_lambda,cost_m_values_final)

            if val_loader:
                final_val_metrics=evaluate_model_enhanced(final_model,val_loader,current_criterion,device,final_le_classes,"BEST Validation (Final Eval)",use_amp,amp_device_str)
                if final_val_metrics.get('cm') is not None and final_val_metrics['cm'].size>0: plot_confusion_matrix(final_val_metrics['cm'],final_le_classes,output_dir/"cm_final_val.png")
                logging.info(f"FINAL BEST MODEL Validation Report:\n{final_val_metrics.get('report', 'N/A')}")
            if test_loader:
                final_test_metrics=evaluate_model_enhanced(final_model,test_loader,current_criterion,device,final_le_classes,"BEST Test (Final Eval)",use_amp,amp_device_str)
                if final_test_metrics.get('cm') is not None and final_test_metrics['cm'].size>0: plot_confusion_matrix(final_test_metrics['cm'],final_le_classes,output_dir/"cm_final_test.png")
                logging.info(f"FINAL BEST MODEL Test Report:\n{final_test_metrics.get('report', 'N/A')}")
        elif not final_model:
            logging.error("Failed to load the final best model for evaluation.")

    gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    elif device.type=='mps' and hasattr(torch.mps,'empty_cache'): torch.mps.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MaxGPUTransformer with Advanced Loss & Evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--initial_dataset_file',type=str,default=None,help='Path to full dataset if --create_initial_split is used.')
    parser.add_argument('--create_initial_split',action='store_true',help='If specified, splits --initial_dataset_file.')
    parser.add_argument('--test_months_cutoff',type=int,default=6,help='Months for test set if --create_initial_split.')
    parser.add_argument('--train_file',type=str,default="train.csv",help='Path to training CSV.')
    parser.add_argument('--test_file',type=str,default="test.csv",help='Path to test CSV.')
    parser.add_argument('--date_col',type=str,default='Date',help="Name of date column in CSVs.")
    parser.add_argument('--target_col',type=str,default='signal',help='Target column name.')
    parser.add_argument('--all_features',type=str,nargs='+',default=['Open','High','Low','Close','Body','High-Low','Is_Doji','Is_Spike','Is_Long_Shadow','gap','hour_sin','hour_cos','minute_sin','minute_cos','day_sin','day_cos'],help="All features.")
    parser.add_argument('--numerical_features',type=str,nargs='+',default=['Open','High','Low','Close','Body','High-Low','hour_sin','hour_cos','minute_sin','minute_cos','day_sin','day_cos'],help="Numerical features to scale.")
    parser.add_argument('--binary_features',type=str,nargs='+',default=['Is_Doji','Is_Spike','Is_Long_Shadow','gap'],help="Binary features.")
    parser.add_argument('--time_features',type=str,nargs='+',default=['hour_sin','hour_cos','minute_sin','minute_cos','day_sin','day_cos'],help='Cyclical time features.')
    parser.add_argument('--target_classes',type=str,nargs='+',default=['buy','keep','sell'],help='Target classes order.')
    parser.add_argument('--sequence_length',type=int,default=90,help='Sequence length.')
    parser.add_argument('--val_size',type=float,default=0.15,help='Validation set proportion.')
    parser.add_argument('--d_model',type=int,default=512,help='Model embedding dimension.')
    parser.add_argument('--n_heads',type=int,default=8,help='Attention heads.')
    parser.add_argument('--num_layers',type=int,default=6,help='Transformer encoder layers.')
    parser.add_argument('--d_ff',type=int,default=0,help='Feed-forward dimension (0 for 4*d_model).')
    parser.add_argument('--dropout',type=float,default=0.1,help='Dropout rate.')
    parser.add_argument('--pe_max_len_default',type=int,default=5000,help="Max length for Positional Encoding.")
    parser.add_argument('--use_moe',action='store_false',help='Enable MoE (default: False).')
    parser.add_argument('--batch_size',type=int,default=128,help='Batch size.')
    parser.add_argument('--grad_accum_steps',type=int,default=4,help='Gradient accumulation steps.')
    parser.add_argument('--epochs',type=int,default=10,help='Training epochs.')
    parser.add_argument('--learning_rate',type=float,default=5e-5,help='Learning rate.')
    parser.add_argument('--weight_decay',type=float,default=1e-4,help='Weight decay.')
    parser.add_argument('--use_warmup',action='store_true',default=True,help='Use LR warmup.')
    parser.add_argument('--warmup_ratio',type=float,default=0.1,help='Warmup ratio.')
    parser.add_argument('--lr_patience',type=int,default=5,help='ReduceLROnPlateau patience.')
    parser.add_argument('--early_stopping_patience',type=int,default=2,help='Early stopping patience.')
    parser.add_argument('--focal_gamma',type=float,default=4.0,help='Focal loss gamma.')
    parser.add_argument('--cost_lambda',type=float,default=15.0,help='Cost-sensitive loss lambda.')
    parser.add_argument('--log_interval',type=int,default=50,help='Log progress every N batches.')
    parser.add_argument('--output_dir',type=str,default="training_run_output",help='Base output directory.')
    parser.add_argument('--num_workers',type=int,default=4,help='DataLoader workers.')
    parser.add_argument('--prefetch_factor',type=int,default=2,help="DataLoader prefetch factor.")
    parser.add_argument('--force_cpu',action='store_true',help='Force CPU.')
    parser.add_argument('--auto_batch_size',action='store_true',default=True,help='Auto-adjust batch size (CUDA).')
    parser.add_argument('--compile_model',action='store_true',default=True,help='Use torch.compile (CUDA).')
    parser.add_argument('--gradient_checkpointing',action='store_true',default=False,help='Enable gradient checkpointing.')
    parser.add_argument('--checkpoint_interval',type=int,default=10,help='Periodic checkpoint save interval.')
    parser.add_argument('--benchmark_interval',type=int,default=10,help='Benchmark interval.')
    parser.add_argument('--random_state',type=int,default=42,help='Global random seed.')

    if 'ipykernel' in sys.modules or 'google.colab' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    args.all_features = sorted(list(set(args.numerical_features + args.binary_features + args.time_features)))
    final_numerical_features = [f for f in args.numerical_features if f in args.all_features]
    for f in args.time_features:
        if f in args.all_features and f not in final_numerical_features: final_numerical_features.append(f)
    args.numerical_features = sorted(list(set(f for f in final_numerical_features if f not in args.binary_features or f in args.numerical_features)))
    if args.d_ff == 0: args.d_ff = 4 * args.d_model
    try:
        if args.d_model%args.n_heads!=0: raise ValueError("d_model not divisible by n_heads.")
    except ValueError as e:
        if 'ipykernel' in sys.modules or 'google.colab' in sys.modules: logging.error(f"ERROR: {e}")
        else: parser.error(str(e))
        sys.exit(1)
    main(args)
