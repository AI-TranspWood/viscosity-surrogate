import json
import os
import sys
from importlib import resources

import numpy as np

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
        epochs: int = defs.DEFAULT_EPOCHS,
        lr: float = defs.DEFAULT_LR,
        batch_size: int = defs.DEFAULT_BATCH_SIZE,

        output_dir: str = "ensemble_output",
    ):
    """Train the XGBoost using the same descriptors as the ChemBERTa model
    and combine the predictions with the two ChemBERTa models."""
    if os.path.exists(output_dir):
        logger.error(f"Output directory '{output_dir}' already exists. Files may be overwritten.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    from .chemberta import train as train_chemberta
    from .chemberta_residuals import train as train_chemberta_residuals
    from .models import get_descriptors
    from .xgboost import train as train_xgboost

    logfile = os.path.join(output_dir, "training.log")
    ensemble_plot_file = os.path.join(output_dir, "ensemble_predictions.png")
    ensemble_config_file = os.path.join(output_dir, "ensemble_config.json")

    lg.add_file_handler(logger, logfile)

    logger.info(f"Running XGBoost training for the ensemble... -------------")
    xgb_res = train_xgboost(
        input_file=input_file,
        train_size=train_size,
        random_state=random_state,
        output_dir=output_dir,
        desc_func=get_descriptors,
        fp_func=dst.get_fingerprint_morgan,
    )
    logger.info(f"Running ChemBERTa training for the ensemble... -------------")
    chem_base_res = train_chemberta(
        input_file=input_file,
        train_size=train_size,
        random_state=random_state,
        model_name=model_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        output_dir=output_dir,
    )
    logger.info(f"Running ChemBERTa Residuals training for the ensemble... -------------")
    chem_res_res = train_chemberta_residuals(
        input_file=input_file,
        train_size=train_size,
        random_state=random_state,
        model_name=model_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    xgb_pred, xgb_mae, xgb_r2 = xgb_res["pred"], xgb_res["mae"], xgb_res["r2"]
    chem_base_pred, chem_base_mae, chem_base_r2 = chem_base_res["pred"], chem_base_res["mae"], chem_base_res["r2"]
    chem_res_pred, chem_res_mae, chem_res_r2 = chem_res_res["pred"], chem_res_res["mae"], chem_res_res["r2"]

    xgb_idx = xgb_res["test_idx"]
    chem_base_idx = chem_base_res["test_idx"]
    chem_res_idx = chem_res_res["test_idx"]

    if np.any(xgb_idx != chem_base_idx) or np.any(xgb_idx != chem_res_idx):
        logger.error("The test indices of the three models do not match. Please ensure they are compatible.")
        sys.exit(1)
    test_y = xgb_res["test_y"]

    best_mae  = float('inf')
    best_combo = None
    best_pred  = None
    best_r2   = None
    best_rmse = None
    results   = []

    msg = [""]
    msg.append("="*55)
    msg.append("ENSEMBLE: Testing weight combinations...")
    msg.append("="*55)
    msg.append(f"{'XGB':>6} {'CHEMBERTA_BASE':>6} {'CHEMBERTA_RES':>6} {'MAE':>8} {'R²':>8}")
    msg.append("-" * 40)

    for w1 in np.arange(0.3, 0.8, 0.1):
        for w2 in np.arange(0.1, 0.5, 0.1):
            w3 = round(1 - w1 - w2, 1)
            if w3 < 0.05:
                continue
            pred = w1*xgb_pred + w2*chem_base_pred + w3*chem_res_pred
            mae, rmse, r2 = dst.evaluate_model(test_y, pred)
            results.append((mae, r2, w1, w2, w3, pred))
            if mae < best_mae:
                best_mae   = mae
                best_r2    = r2
                best_rmse  = rmse
                best_combo = (w1, w2, w3)
                best_pred  = pred
            msg.append(f"{w1:>6.1f} {w2:>6.1f} {w3:>6.1f}  {mae:>8.4f} {r2:>8.4f}")

    xgb_mae, _, xgb_r2 = dst.evaluate_model(test_y, xgb_pred)
    chem_base_mae, _, chem_base_r2 = dst.evaluate_model(test_y, chem_base_pred)
    chem_res_mae, _, chem_res_r2 = dst.evaluate_model(test_y, chem_res_pred)


    msg.append(f"{'='*55}")
    msg.append(f"FINAL SUMMARY")
    msg.append(f"{'='*55}")
    msg.append(f"XGBoost:        MAE={xgb_mae:.4f}  R²={xgb_r2:.4f}")
    msg.append(f"ChemBERTa Base: MAE={chem_base_mae:.4f}  R²={chem_base_r2:.4f}")
    msg.append(f"ChemBERTa Residual:   MAE={chem_res_mae:.4f}  R²={chem_res_r2:.4f}")
    msg.append(
        f"Triple Ensemble:MAE={best_mae:.4f}  R²={best_r2:.4f}  "
        f"({best_combo[0]:.1f} XGB + "
        f"{best_combo[1]:.1f} ChemBERTa Base + "
        f"{best_combo[2]:.1f} ChemBERTa Residual)"
    )

    logger.info("\n".join(msg))

    dst.plot_predictions(test_y, best_pred, best_mae, best_rmse, best_r2, output_file=ensemble_plot_file)

    ensemble_config = {
        "xgb_weight": best_combo[0],
        "chemberta_base_weight": best_combo[1],
        "chemberta_residual_weight": best_combo[2],
    }
    with open(ensemble_config_file, "w") as f:
        json.dump(ensemble_config, f, indent=4)
    logger.info(f"Ensemble configuration saved to {ensemble_config_file}")


def inference(
        input_file: str,
        xgb_model_file: str = None,
        chemberta_base_model_file: str = None,
        chemberta_residuals_model_file: str = None,
        ensemble_config_file: str = None,
        predictions_file: str = "viscosity_inference.csv"
    ):
    """Run the inference using the 3 trained models and give the best combination of weights for the ensemble."""
    if ensemble_config_file is None:
        ensemble_config_file = resources.files("aitw_viscosity_surrogate") / "ensemble_config.json"

    with open(ensemble_config_file, "r") as f:
        ensemble_config = json.load(f)

    df = dst.load_data(input_file)
    smiles_list = df["SMILES"].tolist()

    # Defer heavy imports as late as possible to avoid unnecessary overhead when the CLI is invoked
    import torch

    from .chemberta import load_from_file as load_chemberta_model
    from .chemberta_residuals import \
        load_from_file as load_chemberta_residuals_model
    from .models import get_descriptors
    from .xgboost import load_from_file as load_xgb_model

    model_xgb = load_xgb_model(xgb_model_file)
    model_chemberta_base, tokenizer = load_chemberta_model(chemberta_base_model_file)
    model_chemberta_residuals, _ = load_chemberta_residuals_model(chemberta_residuals_model_file)

    tokens = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt", max_length=128)
    decs = np.array([get_descriptors(s) for s in smiles_list])
    desc_t = torch.FloatTensor(decs)

    X = dst.build_xgboost_features(
        df,
        desc_func=get_descriptors,
        fp_func=dst.get_fingerprint_morgan,
    )

    xgb_log_v = model_xgb.predict(X)
    with torch.no_grad():
        chemberta_base_log_v = model_chemberta_base(
            tokens["input_ids"], tokens["attention_mask"]
        ).squeeze().numpy()
        chemberta_res_log_v = model_chemberta_residuals(
            tokens["input_ids"], tokens["attention_mask"], desc_t
        ).squeeze().numpy()

    w_xgb = ensemble_config["xgb_weight"]
    w_chemberta_base = ensemble_config["chemberta_base_weight"]
    w_chemberta_res = ensemble_config["chemberta_residual_weight"]
    ensemble_log_v = (
        w_xgb * xgb_log_v +
        w_chemberta_base * chemberta_base_log_v +
        w_chemberta_res * chemberta_res_log_v
    )

    xgb_visc = np.round(10 ** xgb_log_v, 4)
    chemberta_base_visc = np.round(10 ** chemberta_base_log_v, 4)
    chemberta_res_visc = np.round(10 ** chemberta_res_log_v, 4)
    ensemble_visc = np.round(10 ** ensemble_log_v, 4)

    df["predicted_log_visc_xgb"] = xgb_log_v
    df["predicted_mPas_xgb"] = xgb_visc
    df["predicted_log_visc_chemberta_base"] = chemberta_base_log_v
    df["predicted_mPas_chemberta_base"] = chemberta_base_visc
    df["predicted_log_visc_chemberta_residuals"] = chemberta_res_log_v
    df["predicted_mPas_chemberta_residuals"] = chemberta_res_visc
    df["predicted_log_visc_ensemble"] = ensemble_log_v
    df["predicted_mPas_ensemble"] = ensemble_visc

    df.to_csv(predictions_file, index=False)
    logger.info(f"Saved predictions to {predictions_file}")
