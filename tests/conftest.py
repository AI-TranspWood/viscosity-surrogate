"""Fixtures for the aitw-viscosity-surrogate tests."""
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def dataset_file_path():
    """Fixture to provide the path to the dataset file."""
    return "examples/dataset.csv"


@pytest.fixture(scope="session")
def dataset(dataset_file_path):
    """Fixture to provide the path to the dataset."""
    return pd.read_csv(dataset_file_path)
