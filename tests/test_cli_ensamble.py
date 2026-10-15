"""Test the CLI of the aitw-viscosity-surrogate package."""
import os

import pandas as pd
from click.testing import CliRunner

# def test_cli_ensemble_existing_output_dir(dataset_file_path, tmpdir, caplog):
#     """Test the CLI train command with an existing output directory."""
#     from aitw_viscosity_surrogate.cli import cli

#     tmp_output_dir = tmpdir / "output"
#     os.makedirs(tmp_output_dir)

#     runner = CliRunner()
#     with caplog.at_level("WARNING"):
#         result = runner.invoke(cli, [
#             'chemberta-residuals', 'train',
#             '--input-file', dataset_file_path,
#             '--output-dir', str(tmp_output_dir),
#         ])

#     assert result.exit_code == 1
#     assert 'already exists' in caplog.text


def test_cli_ensemble_train_and_inference(dataset_file_path, tmpdir):
    """Test the CLI train command."""
    from aitw_viscosity_surrogate.cli import cli

    tmp_output_dir = tmpdir / "output"

    runner = CliRunner()
    xgb_weights_file = os.path.join(tmp_output_dir, "xgboost_model.json")
    chemberta_base_weights_file = os.path.join(tmp_output_dir, "chemberta_model_base.pt")
    chemberta_residuals_weights_file = os.path.join(tmp_output_dir, "chemberta_model_residuals.pt")
    ensemble_config_file = os.path.join(tmp_output_dir, "ensemble_config.json")
    result = runner.invoke(cli, [
        'ensemble', 'train',
        '--input-file', dataset_file_path,
        '--output-dir', str(tmp_output_dir),
        '--epochs', '1',
        '--batch-size', '4',
    ])

    assert result.exit_code == 0
    assert os.path.exists(xgb_weights_file)
    assert os.path.exists(chemberta_base_weights_file)
    assert os.path.exists(chemberta_residuals_weights_file)
    assert os.path.exists(ensemble_config_file)

    inference_results = os.path.join(tmpdir, 'inference_results.csv')
    result = runner.invoke(cli, [
        'ensemble', 'inference',
        '--input-file', dataset_file_path,
        '--xgb-model-file', xgb_weights_file,
        '--chemberta-base-model-file', chemberta_base_weights_file,
        '--chemberta-residuals-model-file', chemberta_residuals_weights_file,
        '--ensemble-config-file', ensemble_config_file,
        '--output-file', inference_results,
    ])

    if result.exit_code != 0:
        print(result.output)
        print(result.exception)
        raise result.exception
    assert result.exit_code == 0
    assert os.path.exists(inference_results)

    inference_results_df = pd.read_csv(inference_results)
    assert all(col in inference_results_df.columns for col in [
        'compound_name', 'SMILES',
        'predicted_log_visc_xgb', 'predicted_mPas_xgb',
        'predicted_log_visc_chemberta_base', 'predicted_mPas_chemberta_base',
        'predicted_log_visc_chemberta_residuals', 'predicted_mPas_chemberta_residuals',
        'predicted_log_visc_ensemble', 'predicted_mPas_ensemble'
    ])

def test_cli_ensemble_inference_with_default_weights(dataset_file_path, tmpdir):
    """Test the CLI inference command with default weights."""
    from aitw_viscosity_surrogate.cli import cli

    runner = CliRunner()
    inference_results = os.path.join(tmpdir, 'inference_results.csv')
    result = runner.invoke(cli, [
        'ensemble', 'inference',
        '--input-file', dataset_file_path,
        '--output-file', inference_results,
    ])

    if result.exit_code != 0:
        print(result.output)
        print(result.exception)
        raise result.exception
    assert result.exit_code == 0
    assert os.path.exists(inference_results)

    inference_results_df = pd.read_csv(inference_results)
    assert all(col in inference_results_df.columns for col in [
        'compound_name', 'SMILES',
        'predicted_log_visc_xgb', 'predicted_mPas_xgb',
        'predicted_log_visc_chemberta_base', 'predicted_mPas_chemberta_base',
        'predicted_log_visc_chemberta_residuals', 'predicted_mPas_chemberta_residuals',
        'predicted_log_visc_ensemble', 'predicted_mPas_ensemble'
    ])
