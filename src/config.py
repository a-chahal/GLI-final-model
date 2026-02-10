"""
GLI-PLAPT Configuration — All hyperparameters, paths, and constants.
"""

import os
from dataclasses import dataclass, field
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
EMBED_CACHE_DIR = os.path.join(OUTPUT_DIR, "embedding_cache")

SEED = 42
MULTI_SEEDS = [42, 123, 456, 789, 1024]


@dataclass
class DataPaths:
    bindingdb: str = os.path.join(DATA_DIR, "bindingdb_200k.csv")
    bindingdb_sequences: str = os.path.join(DATA_DIR, "uniprot_sequences_200k.fasta")
    zf_combined: str = os.path.join(DATA_DIR, "zf_training_combined.csv")
    zf_binding_only: str = os.path.join(DATA_DIR, "zf_training_binding_only.csv")
    gli_inhibitors: str = os.path.join(DATA_DIR, "gli_inhibitors.csv")
    gli_finetuning: str = os.path.join(DATA_DIR, "gli_finetuning_data.csv")
    gli_finetuning_augmented: str = os.path.join(DATA_DIR, "gli_finetuning_data_augmented.csv")
    negatives: str = os.path.join(DATA_DIR, "negatives.csv")
    augmentation_mapping: str = os.path.join(DATA_DIR, "augmentation_mapping.csv")
    gli1_fasta: str = os.path.join(DATA_DIR, "gli1_sequence.fasta")
    gli_binding_data: str = os.path.join(DATA_DIR, "gli_binding_data.csv")
    master_targets: str = os.path.join(DATA_DIR, "master_targets.csv")


# Negative sources to INCLUDE (user decision: SMO inhibitors + structural only)
NEGATIVE_SOURCES_KEEP = [
    "ChEMBL_SMO_inhibitor", "GANT61_inactive_analog", "random_molecule",
    "inactive_BAS07019774_analog_missing_pyrrolidine",
]


@dataclass
class EncoderConfig:
    # ESM-2 650M
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D"
    esm2_embed_dim: int = 1280
    esm2_max_length: int = 1024  # GLI1 is ~1232 AA; truncate safely

    # ProtBERT (baseline)
    protbert_model_name: str = "Rostlab/prot_bert"
    protbert_embed_dim: int = 1024
    protbert_max_length: int = 3200

    # ChemBERTa (shared)
    chemberta_model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    chemberta_embed_dim: int = 768
    chemberta_max_length: int = 278


@dataclass
class PredictionHeadConfig:
    # Branching architecture
    protein_branch_out: int = 256
    ligand_branch_out: int = 256
    fusion_hidden_1: int = 256
    fusion_hidden_2: int = 128
    branch_dropout: float = 0.2
    fusion_dropout: float = 0.3

    # MC Dropout inference
    mc_samples: int = 50

    # Morgan fingerprint hybrid (Phase 2C)
    use_morgan_fp: bool = True
    morgan_fp_bits: int = 2048
    morgan_fp_radius: int = 2
    morgan_fp_hidden: int = 128


@dataclass
class PretrainConfig:
    """Stage 1: BindingDB pretraining."""
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    patience: int = 5
    val_split: float = 0.1
    cosine_T0: int = 5
    # Binarization thresholds (nM)
    active_threshold_nM: float = 1000.0    # <= 1000 nM → active
    inactive_threshold_nM: float = 5000.0   # > 5000 nM → inactive (BindingDB capped at 10μM)
    smiles_augment_per_epoch: int = 1


@dataclass
class DomainAdaptConfig:
    """Stage 2: ZF domain adaptation."""
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 10
    val_split: float = 0.2
    cosine_T0: int = 10


@dataclass
class GLIFinetuneConfig:
    """Stage 3: GLI-specific LOOCV fine-tuning."""
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 1e-5
    epochs: int = 30
    patience: int = 7
    smiles_augment_per_positive: int = 10
    cosine_T0: int = 5

    # Focal loss (Phase 2B)
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75

    # Asymmetric augmentation (Phase 2D)
    use_asymmetric_aug: bool = True
    asym_aug_min: int = 5
    asym_aug_max: int = 25


@dataclass
class Config:
    paths: DataPaths = field(default_factory=DataPaths)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head: PredictionHeadConfig = field(default_factory=PredictionHeadConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    domain_adapt: DomainAdaptConfig = field(default_factory=DomainAdaptConfig)
    gli_finetune: GLIFinetuneConfig = field(default_factory=GLIFinetuneConfig)
    seed: int = SEED
    use_esm2: bool = True  # False = baseline ProtBERT
