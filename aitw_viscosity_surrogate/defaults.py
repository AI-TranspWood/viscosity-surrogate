"""Default configuration values for the viscosity surrogate model."""

DEFAULT_MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
DEFAULT_EPOCHS     = 1000
DEFAULT_LR         = 2e-5
DEFAULT_BATCH_SIZE = 16
DEFAULT_DROPOUT    = 0.1
DEFAULT_HIDDEN     = 256

RDKIT_DESC_NAMES = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "TPSA",
    "NumAromaticRings",
    "NumAliphaticRings",
    "HeavyAtomCount",
    "FractionCSP3",
]
DEFAULT_N_DESC = len(RDKIT_DESC_NAMES)  # 10
DEFAULT_CHEMBERTA_DIM = 384
DEFAULT_COMBINED_DIM  = DEFAULT_CHEMBERTA_DIM + DEFAULT_N_DESC  # 394
DEFAULT_HIDDEN_DIM    = 512   # 
DEFAULT_N_RES_BLOCKS  = 2    #
