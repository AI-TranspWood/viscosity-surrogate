import json
import os
import sys
from importlib import resources

import numpy as np
import pandas as pd

from . import dataset as dst
from . import defaults as defs
from . import loggers as lg

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

logger = lg.MAIN_LOGGER


def train(
        input_file: str,
        train_size: float = 0.8,
        random_state: int = 42,
        model_name: str = defs.DEFAULT_MODEL_NAME,
        hidden: int = defs.DEFAULT_HIDDEN,
        dropout: float = defs.DEFAULT_DROPOUT,
        epochs: int = defs.DEFAULT_EPOCHS,
        lr: float = defs.DEFAULT_LR,
        batch_size: int = defs.DEFAULT_BATCH_SIZE,
        output_dir: str = "chemberta_baseline_output",
    ):
    """Fine-tune ChemBERTa on the dataset."""
    if os.path.exists(output_dir):
        logger.error(f"Output directory '{output_dir}' already exists. Files may be overwritten.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    # Defer heavy imports as late as possible to avoid unnecessary overhead when the CLI is invoked
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer

    from .models import ChemBERTaViscosity

    logfile = os.path.join(output_dir, "training.log")
    plot_file = os.path.join(output_dir, "viscosity_performance.png")
    indexes_file = os.path.join(output_dir, "train_test_indexes.json")
    predictions_file = os.path.join(output_dir, "viscosity_predictions.csv")
    model_file = os.path.join(output_dir, "chemberta_model_base.pt")

    lg.add_file_handler(logger, logfile)
    # lg.set_console_level(logger, "INFO")

    # 1. Load
    df = pd.read_csv(input_file)

    # 2. Split
    train_idx, test_idx = dst.split_dataset(df, train_size=train_size, random_state=random_state)
    train_smiles = df["SMILES"].iloc[train_idx].tolist()
    test_smiles  = df["SMILES"].iloc[test_idx].tolist()
    train_y      = df["log_visc_exp"].iloc[train_idx].values.astype(np.float32)
    test_y       = df["log_visc_exp"].iloc[test_idx].values.astype(np.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ChemBERTaViscosity(hidden=hidden, dropout=dropout)

    logger.info(f"Loaded ChemBERTa model '{model_name}' with {sum(p.numel() for p in model.parameters()):,} parameters.")

    train_tokens = tokenizer(train_smiles, padding=True, truncation=True, return_tensors="pt")
    test_tokens = tokenizer(test_smiles, padding=True, truncation=True, return_tensors="pt")
    train_y_t = torch.FloatTensor(train_y)

    logger.info(f"Training set: {len(train_idx)} molecules")
    logger.info(f"Testing set:  {len(test_idx)} molecules")

    logger.info(f"Starting fine-tuning for {epochs} epochs with learning rate {lr} and batch size {batch_size}...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches  = 0
        for i in range(0, len(train_smiles), batch_size):
            logger.debug(
                f"Epoch {epoch+1}/{epochs}, Batch {n_batches+1}: "
                f"Training on samples {i} to {min(i+batch_size, len(train_smiles))}..."
            )
            optimizer.zero_grad()
            pred = model(
                train_tokens['input_ids'][i:i+batch_size],
                train_tokens['attention_mask'][i:i+batch_size]).squeeze()
            loss = loss_fn(pred, train_y_t[i:i+batch_size])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(
                    test_tokens['input_ids'],
                    test_tokens['attention_mask']).squeeze().numpy()
            mae, rmse, r2 = dst.evaluate_model(test_y, test_pred)
            logger.info(
                f"  Epoch {epoch+1:>3}/{epochs} | "
                f"Loss: {total_loss/n_batches:.4f} | "
                f"Test MAE: {mae:.4f}"
            )
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(
            test_tokens['input_ids'],
            test_tokens['attention_mask']
        ).squeeze().numpy()

    mae, rmse, r2 = dst.evaluate_model(test_y, test_pred)
    logger.info(f"\nFinal Evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    dst.plot_predictions(test_y, test_pred, mae, rmse, r2, output_file=plot_file)
    dst.print_predictions(df, test_idx, test_pred)
    dst.save_predictions(df, test_idx, test_y, test_pred, predictions_file)

    # Save model
    torch.save(model.state_dict(), model_file)
    indexes = {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()}
    with open(indexes_file, "w") as f:
        json.dump(indexes, f, indent=4)
    logger.info(f"Model saved to '{model_file}'")
    logger.info(f"Train/test indexes saved to '{indexes_file}'")


def inference(
        input_file: str,
        model_file: str = None,
        predictions_file: str = "viscosity_inference.csv",
    ):
    """Run inference using a pre-trained ChemBERTa model."""
    if model_file is None:
        with resources.path("aitw_viscosity_surrogate", "chemberta_model_base.pt") as path:
            model_file = str(path)
            logger.debug(f"No model file provided. Using default model: '{model_file}'")

    if os.path.exists(predictions_file):
        logger.error(f"Predictions file '{predictions_file}' already exists. Please specify a different output file.")
        sys.exit(1)
    basedir = os.path.dirname(predictions_file)
    if basedir:
        os.makedirs(basedir, exist_ok=True)

    # Defer heavy imports as late as possible to avoid unnecessary overhead when the CLI is invoked
    import torch
    from transformers import AutoTokenizer

    from .models import ChemBERTaViscosity

    df = pd.read_csv(input_file)
    smiles_list = df["SMILES"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(defs.DEFAULT_MODEL_NAME)
    model = ChemBERTaViscosity()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    tokens = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        log_v = model(tokens['input_ids'], tokens['attention_mask']).squeeze().numpy()
    visc      = np.round(10**log_v, 4)

    df["predicted_log_visc"] = log_v
    df["predicted_mPas"]     = visc
    if "visc_mPas_exp" in df.columns:
        actual = df["visc_mPas_exp"].values
        error_pct = np.abs(visc - actual) / actual * 100
        df["error_%"] = error_pct

    df.to_csv(predictions_file, index=False)
    logger.info(f"Predictions saved to '{predictions_file}'")
