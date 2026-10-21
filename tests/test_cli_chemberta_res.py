"""Test the CLI of the aitw-viscosity-surrogate package."""
import os

import pandas as pd
from click.testing import CliRunner

# def test_cli_chemberta_res_existing_output_dir(dataset_file_path, tmpdir, caplog):
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


def test_cli_chemberta_res_train_and_inference(dataset_file_path, tmpdir):
    """Test the CLI train command."""
    from aitw_viscosity_surrogate.cli import cli

    tmp_output_dir = tmpdir / "output"

    runner = CliRunner()
    output_weights_file = os.path.join(tmp_output_dir, "chemberta_model_residuals.pt")
    result = runner.invoke(cli, [
        'chemberta-residuals', 'train',
        '--input-file', dataset_file_path,
        '--output-dir', str(tmp_output_dir),
        '--epochs', '1',
        '--batch-size', '4',
    ])

    assert result.exit_code == 0
    assert os.path.exists(output_weights_file)

    inference_results = os.path.join(tmpdir, 'inference_results.csv')
    result = runner.invoke(cli, [
        'chemberta-residuals', 'inference',
        '--input-file', dataset_file_path,
        '--model-file', output_weights_file,
        '--output-file', inference_results,
    ])

    assert result.exit_code == 0
    assert os.path.exists(inference_results)

    inference_results_df = pd.read_csv(inference_results)
    assert all(col in inference_results_df.columns for col in [
        'compound_name', 'SMILES', 'predicted_log_visc', 'predicted_mPas'
    ])

def test_cli_chemberta_res_inference_with_default_weights(dataset_file_path, tmpdir):
    """Test the CLI inference command with default weights."""
    from aitw_viscosity_surrogate.cli import cli

    runner = CliRunner()
    inference_results = os.path.join(tmpdir, 'inference_results.csv')
    result = runner.invoke(cli, [
        'chemberta-residuals', 'inference',
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
        'compound_name', 'SMILES', 'predicted_log_visc', 'predicted_mPas'
    ])
