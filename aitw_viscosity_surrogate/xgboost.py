import json
import os
from importlib import resources

import numpy as np
import xgboost as xgb

from . import dataset as dst
from . import loggers as lg

logger = lg.MAIN_LOGGER


def train(
        input_file: str,
        train_size: float = 0.8,
        random_state: int = 42,
        output_dir: str = "xgboost_baseline_output",
        desc_func=dst.get_all_descriptors,
        fp_func=dst.get_fingerprint_morgan,
        compatibility: bool = False,
    ):
    """Train the XGBoost model on the dataset."""
    if os.path.exists(output_dir):
        logger.warning(f"Output directory '{output_dir}' already exists. Files may be overwritten.")
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(random_state)

    logfile = os.path.join(output_dir, "xgboost_training.log")
    plot_file = os.path.join(output_dir, "xgboost_performance.png")
    predictions_file = os.path.join(output_dir, "xgboost_predictions.csv")
    model_file = os.path.join(output_dir, "xgboost_model.json")
    indexes_file = os.path.join(output_dir, "xgboost_indexes.json")

    lg.add_file_handler(logger, logfile)

    # 1. Load
    df = dst.load_data(input_file)

    # 2. Build features
    X = dst.build_xgboost_features(df, desc_func=desc_func, fp_func=fp_func, compatibility=compatibility)
    y = df["log_visc_exp"].values

    # 3. Split
    train_idx, test_idx = dst.split_dataset(X, train_size=train_size, random_state=random_state)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    logger.info(f"Training set: {len(train_idx)} molecules")
    logger.info(f"Testing set:  {len(test_idx)} molecules")

    # 4. Train
    logger.info("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators     = 500,
        learning_rate    = 0.1,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = random_state,
        n_jobs           = -1,
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred       = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    mae_tr, rmse_tr, r2_tr = dst.evaluate_model(y_train, y_pred_train)
    mae_te, rmse_te, r2_te = dst.evaluate_model(y_test, y_pred)

    logger.info("=" * 55)
    logger.info(f"Train: MAE={mae_tr:.4f}  RMSE={rmse_tr:.4f}  R²={r2_tr:.4f}")
    logger.info(f"Test:  MAE={mae_te:.4f}  RMSE={rmse_te:.4f}  R²={r2_te:.4f}")
    gap  = r2_tr - r2_te
    flag = "low" if gap < 0.05 else "moderate" if gap < 0.10 else "high"
    logger.info(f"Overfit gap: {gap:.4f}  ({flag})")
    
    logger.info("=" * 55)

    dst.plot_predictions(y_test, y_pred, mae_te, rmse_te, r2_te, output_file=plot_file)
    dst.print_predictions(df, test_idx, y_pred)
    dst.save_predictions(df, test_idx, y_test, y_pred, predictions_file)

    # 9. Save model
    model.save_model(model_file)
    logger.info(f"Model saved to {model_file}")

    indexes = {
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist()
    }
    with open(indexes_file, "w") as f:
        json.dump(indexes, f, indent=4)
    logger.info(f"Train/test indexes saved to {indexes_file}")

    res = {
        "mae": mae_te,
        "rmse": rmse_te,
        "r2": r2_te,
        "model": model,
        "pred": y_pred,
        "test_y": y_test,
        "test_idx": test_idx,
    }
    return res


def load_from_file(model_file: str):
    """Load a pre-trained XGBoost model from a file."""
    if model_file is None:
        model_file = resources.files("aitw_viscosity_surrogate") / "xgboost_model.json"

    model = xgb.XGBRegressor()
    model.load_model(model_file)

    logger.info(f"Loaded XGBoost model from {model_file}")

    return model

def inference(
        input_file: str,
        model_file: str = None,
        predictions_file: str = "viscosity_inference.csv",
        compatibility: bool = False,
    ):
    """Run inference using a pre-trained XGBoost model."""
    if os.path.exists(predictions_file):
        logger.error(f"Predictions file '{predictions_file}' already exists. Files may be overwritten.")
    basedir = os.path.dirname(predictions_file)
    if basedir:
        os.makedirs(basedir, exist_ok=True)
        
    # 1. Load data
    df = dst.load_data(input_file)

    # 2. Build features
    X = dst.build_xgboost_features(df, compatibility=compatibility,)

    # 3. Load model
    model = load_from_file(model_file)

    # 4. Predict
    y_pred = model.predict(X)

    log_v = y_pred
    visc = np.round(10 ** log_v, 4)

    # 5. Save Predictions to CSV
    out = df[["compound_name", "SMILES"]].copy()
    out["predicted_log_visc"] = y_pred
    out["predicted_mPas"]     = np.round(10 ** y_pred, 4)

    if "visc_mPas_exp" in df.columns:
        actual = df["visc_mPas_exp"].values
        error_pct = np.abs(visc - actual) / actual * 100
        out["error_%"] = error_pct

    out.to_csv(predictions_file, index=False)
    logger.info(f"Saved predictions to {predictions_file}")
