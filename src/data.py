"""
GLI-PLAPT Data — Loading, preprocessing, augmentation, and PyTorch datasets.
"""

import os
import re
import csv
import logging
import hashlib
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.config import (
    Config, DataPaths, NEGATIVE_SOURCES_KEEP, EMBED_CACHE_DIR
)

# ---------------------------------------------------------------------------
# SMILES Augmentation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Morgan Fingerprints and Tanimoto Similarity (Phase 2)
# ---------------------------------------------------------------------------

def compute_morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Compute Morgan (ECFP) fingerprint as a binary numpy array."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return np.zeros(n_bits, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


def compute_morgan_fps_batch(smiles_list: List[str], n_bits: int = 2048,
                             radius: int = 2) -> torch.Tensor:
    """Compute Morgan fingerprints for a list of SMILES → (N, n_bits) float tensor."""
    fps = [compute_morgan_fp(s, n_bits, radius) for s in smiles_list]
    return torch.tensor(np.stack(fps), dtype=torch.float32)


def compute_tanimoto_matrix(smiles_list: List[str], radius: int = 2,
                            n_bits: int = 2048) -> np.ndarray:
    """Compute pairwise Tanimoto similarity matrix using Morgan fingerprints."""
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError:
        n = len(smiles_list)
        return np.eye(n)

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles("C"), radius, nBits=n_bits))

    n = len(fps)
    sim_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    return sim_matrix


def compute_asymmetric_aug_counts(smiles_list: List[str],
                                  base_aug: int = 10,
                                  min_aug: int = 5,
                                  max_aug: int = 25) -> List[int]:
    """Compute per-compound augmentation counts inversely weighted by similarity.

    Structurally isolated compounds get more augmentation (up to max_aug);
    clustered compounds get less (down to min_aug). This rebalances the
    effective training distribution to give the model more diverse views
    of hard-to-predict compounds.
    """
    sim_matrix = compute_tanimoto_matrix(smiles_list)
    n = len(smiles_list)
    aug_counts = []

    for i in range(n):
        # Mean similarity to all OTHER compounds
        others = [sim_matrix[i, j] for j in range(n) if j != i]
        mean_sim = np.mean(others) if others else 0.0

        # Inversely map: high similarity → low aug, low similarity → high aug
        # Linear interpolation: sim=0 → max_aug, sim=1 → min_aug
        aug = int(max_aug - (max_aug - min_aug) * mean_sim)
        aug = max(min_aug, min(max_aug, aug))
        aug_counts.append(aug)

    logging.info(f"  Asymmetric augmentation counts: {dict(zip(smiles_list[:3], aug_counts[:3]))}...")
    return aug_counts


def randomize_smiles(smiles: str, n_augments: int = 1) -> List[str]:
    """Generate randomized (non-canonical) SMILES representations.

    Uses RDKit to create equivalent but textually different SMILES.
    Falls back to returning the original if RDKit fails.
    """
    try:
        from rdkit import Chem
    except ImportError:
        logging.warning("RDKit not available; returning original SMILES for augmentation.")
        return [smiles] * n_augments

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles] * n_augments

    augmented = set()
    attempts = 0
    max_attempts = n_augments * 10
    while len(augmented) < n_augments and attempts < max_attempts:
        aug = Chem.MolToSmiles(mol, doRandom=True)
        if aug and aug != smiles:
            augmented.add(aug)
        attempts += 1

    result = list(augmented)
    # Pad with original if we couldn't generate enough unique augmentations
    while len(result) < n_augments:
        result.append(smiles)
    return result[:n_augments]


# ---------------------------------------------------------------------------
# Protein Sequence Handling
# ---------------------------------------------------------------------------

def parse_fasta(filepath: str) -> Dict[str, str]:
    """Parse a FASTA file into {accession: sequence} dict.

    Handles UniProt-style headers like '>sp|P31645|SC6A4_HUMAN ...'
    by extracting the accession (P31645) as the key. Also stores
    the full first-word header as a fallback key.
    """
    sequences = {}
    current_headers = []
    current_seq = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_headers:
                    seq = "".join(current_seq)
                    for h in current_headers:
                        sequences[h] = seq
                full_header = line[1:].split()[0]
                current_headers = [full_header]
                # Extract UniProt accession from sp|ACCESSION|NAME or tr|ACCESSION|NAME
                parts = full_header.split("|")
                if len(parts) >= 2:
                    current_headers.append(parts[1])  # The accession ID
                current_seq = []
            else:
                current_seq.append(line)
    if current_headers:
        seq = "".join(current_seq)
        for h in current_headers:
            sequences[h] = seq
    return sequences


def preprocess_protein_sequence(seq: str) -> str:
    """Preprocess protein sequence for ProtBERT (space-separated, replace rare AAs)."""
    return " ".join(re.sub(r"[UZOB]", "X", seq))


def get_gli1_sequence(fasta_path: str) -> str:
    """Load GLI1 protein sequence from FASTA."""
    seqs = parse_fasta(fasta_path)
    if not seqs:
        raise ValueError(f"No sequences found in {fasta_path}")
    # Return the first (and likely only) sequence
    seq = list(seqs.values())[0]
    logging.info(f"GLI1 sequence loaded: {len(seq)} amino acids")
    return seq


# ---------------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------------

def load_bindingdb(config: Config) -> pd.DataFrame:
    """Load BindingDB pretraining data with proper negative sampling.

    Strategy:
        Positive: all measured protein-ligand pairs (real interactions)
        Negative: randomly shuffled pairs (protein_i, ligand_j where j != i)
        This teaches "real interaction vs random pairing" — the correct
        pretraining objective for downstream binding prediction.
    """
    paths = config.paths
    pt = config.pretrain

    logging.info("Loading BindingDB data...")
    df = pd.read_csv(paths.bindingdb, low_memory=False)
    logging.info(f"  Raw rows: {len(df)}")

    # Drop rows with missing SMILES or affinity
    df = df.dropna(subset=["ligand_smiles", "binding_affinity_nM"])
    df["binding_affinity_nM"] = pd.to_numeric(df["binding_affinity_nM"], errors="coerce")
    df = df.dropna(subset=["binding_affinity_nM"])
    logging.info(f"  After dropping NaN: {len(df)}")

    # Load protein sequences
    logging.info("  Loading protein sequences from FASTA...")
    uniprot_seqs = parse_fasta(paths.bindingdb_sequences)
    logging.info(f"  Loaded {len(uniprot_seqs)} UniProt sequences")

    # Map uniprot_id to sequence
    df["protein_sequence"] = df["uniprot_id"].map(uniprot_seqs)
    df = df.dropna(subset=["protein_sequence"])
    logging.info(f"  After sequence matching: {len(df)}")

    # --- Positive pairs: all measured interactions ---
    positives = df[["ligand_smiles", "protein_sequence"]].copy()
    positives["label"] = 1
    logging.info(f"  Positives (real pairs): {len(positives)}")

    # --- Negative pairs: random protein-ligand shuffling ---
    # For each real pair, create a decoy by pairing the protein with a
    # random ligand from a DIFFERENT protein. This is standard practice
    # (ConPLex, MolTrans, DrugBAN).
    rng = np.random.RandomState(config.seed)
    n_neg = len(positives)  # 1:1 ratio
    shuffled_ligands = positives["ligand_smiles"].values.copy()
    proteins = positives["protein_sequence"].values

    # Shuffle until no ligand is paired with its original protein
    # (simple rejection sampling with fallback)
    for attempt in range(10):
        rng.shuffle(shuffled_ligands)
        same_mask = shuffled_ligands == positives["ligand_smiles"].values
        if same_mask.sum() < 0.01 * n_neg:  # <1% collisions is fine
            break

    negatives = pd.DataFrame({
        "ligand_smiles": shuffled_ligands[:n_neg],
        "protein_sequence": proteins[:n_neg],
        "label": 0,
    })
    logging.info(f"  Negatives (shuffled pairs): {len(negatives)}")

    # Combine
    combined = pd.concat([positives, negatives], ignore_index=True)
    combined = combined.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    logging.info(f"  Combined dataset: {len(combined)} "
                 f"(pos={combined['label'].sum()}, neg={len(combined) - combined['label'].sum()})")

    return combined[["ligand_smiles", "protein_sequence", "label"]].reset_index(drop=True)


def load_zf_data(config: Config) -> pd.DataFrame:
    """Load zinc finger domain adaptation data.

    All ZF data is labeled 'active' in the source file, so we binarize
    using binding affinity thresholds (same as BindingDB):
        Active:   affinity <= 1000 nM  (strong binders)
        Inactive: affinity > 10000 nM  (weak/non-binders)
        Discard:  1000-10000 nM        (ambiguous)
    """
    logging.info("Loading ZF domain adaptation data...")
    df = pd.read_csv(config.paths.zf_combined, low_memory=False)
    logging.info(f"  Raw rows: {len(df)}")

    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if "smiles" in cl and "canonical" in cl:
            col_map[c] = "smiles"
        elif "smiles" in cl:
            col_map[c] = "smiles"
        elif "sequence" in cl or "protein_seq" in cl:
            col_map[c] = "protein_sequence"
        elif "binding_affinity" in cl:
            col_map[c] = "binding_affinity_nM"
    df = df.rename(columns=col_map)

    # Binarize using pchembl_value (more reliable than nM for mixed assay types)
    # Standard medicinal chemistry: pchembl >= 6.0 (≤1μM) = active, < 5.0 (>10μM) = inactive
    if "pchembl_value" in df.columns:
        df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
        df = df.dropna(subset=["smiles", "protein_sequence", "pchembl_value"])
        active = df["pchembl_value"] >= 6.0
        inactive = df["pchembl_value"] < 5.0
    else:
        df["binding_affinity_nM"] = pd.to_numeric(df["binding_affinity_nM"], errors="coerce")
        df = df.dropna(subset=["smiles", "protein_sequence", "binding_affinity_nM"])
        active = df["binding_affinity_nM"] <= config.pretrain.active_threshold_nM
        inactive = df["binding_affinity_nM"] > config.pretrain.inactive_threshold_nM

    df = df[active | inactive].copy()
    df["label"] = active[df.index].astype(int)

    df = df.dropna(subset=["smiles", "protein_sequence", "label"])
    logging.info(f"  After binarization: {len(df)} (active={df['label'].sum()}, "
                 f"inactive={len(df) - df['label'].sum()})")

    return df[["smiles", "protein_sequence", "label"]].reset_index(drop=True)


def load_gli_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Load GLI inhibitors (positives), filtered negatives, and ChEMBL GLI binding data.

    The ChEMBL GLI binding data (gli_binding_data.csv) contains 56 compounds
    with measured IC50 against GLI1/GLI2. We binarize by pchembl_value:
        Active (supplementary positives): pchembl >= 6.0 (IC50 <= 1 uM)
        Inactive (GLI-tested negatives):  pchembl < 5.0  (IC50 > 10 uM)

    Returns:
        positives: DataFrame with SMILES of 11 confirmed GLI binders
        negatives: DataFrame with SMILES of filtered negatives (SMO + structural)
        chembl_gli_pos: DataFrame of ChEMBL GLI actives (supplementary positives)
        chembl_gli_neg: DataFrame of ChEMBL GLI inactives (true GLI negatives)
        gli1_seq: GLI1 protein sequence
    """
    logging.info("Loading GLI data...")

    # Load GLI1 sequence
    gli1_seq = get_gli1_sequence(config.paths.gli1_fasta)

    # Load all 11 confirmed GLI binders
    gli_df = pd.read_csv(config.paths.gli_inhibitors)
    logging.info(f"  GLI inhibitors: {len(gli_df)} compounds")

    # Load and filter negatives
    neg_df = pd.read_csv(config.paths.negatives)
    neg_df = neg_df[neg_df["source"].isin(NEGATIVE_SOURCES_KEEP)]
    logging.info(f"  Filtered negatives: {len(neg_df)} "
                 f"(sources: {dict(neg_df['source'].value_counts())})")

    # Load ChEMBL GLI binding data (supplementary positives + true GLI negatives)
    chembl_gli_pos = pd.DataFrame(columns=["smiles"])
    chembl_gli_neg = pd.DataFrame(columns=["smiles"])
    try:
        gli_bind = pd.read_csv(config.paths.gli_binding_data, low_memory=False)
        gli_bind["pchembl_value"] = pd.to_numeric(gli_bind["pchembl_value"], errors="coerce")
        gli_bind = gli_bind.dropna(subset=["canonical_smiles", "pchembl_value"])

        # Deduplicate by SMILES (keep highest pchembl for same compound)
        gli_bind = gli_bind.sort_values("pchembl_value", ascending=False)
        gli_bind = gli_bind.drop_duplicates(subset=["canonical_smiles"], keep="first")

        # Remove any compounds already in our 11 confirmed inhibitors
        existing_smiles = set(gli_df["smiles"].values) if "smiles" in gli_df.columns else set()
        gli_bind = gli_bind[~gli_bind["canonical_smiles"].isin(existing_smiles)]

        # Binarize
        active = gli_bind[gli_bind["pchembl_value"] >= 6.0]  # IC50 <= 1 uM
        inactive = gli_bind[gli_bind["pchembl_value"] < 5.0]  # IC50 > 10 uM

        chembl_gli_pos = active[["canonical_smiles"]].rename(columns={"canonical_smiles": "smiles"})
        chembl_gli_neg = inactive[["canonical_smiles"]].rename(columns={"canonical_smiles": "smiles"})

        logging.info(f"  ChEMBL GLI actives (supplementary positives): {len(chembl_gli_pos)}")
        logging.info(f"  ChEMBL GLI inactives (true GLI negatives): {len(chembl_gli_neg)}")
    except Exception as e:
        logging.warning(f"  Could not load gli_binding_data.csv: {e}")

    return gli_df, neg_df, chembl_gli_pos, chembl_gli_neg, gli1_seq


# ---------------------------------------------------------------------------
# Embedding Cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Disk-based embedding cache using torch.save/load.

    Caches protein and ligand embeddings to avoid recomputing them.
    """

    def __init__(self, cache_dir: str = EMBED_CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, text: str, model_name: str) -> str:
        h = hashlib.md5(f"{model_name}:{text}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.pt")

    def get(self, text: str, model_name: str) -> Optional[torch.Tensor]:
        path = self._key(text, model_name)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def put(self, text: str, model_name: str, embedding: torch.Tensor):
        path = self._key(text, model_name)
        torch.save(embedding.cpu(), path)

    def has(self, text: str, model_name: str) -> bool:
        return os.path.exists(self._key(text, model_name))


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class ProteinLigandDataset(Dataset):
    """Dataset for protein-ligand pairs with binary labels.

    Stores raw SMILES and protein sequences. Embedding is done externally
    (pre-computed and cached) to keep the dataset simple.
    """

    def __init__(self, smiles: List[str], protein_seqs: List[str],
                 labels: List[int], augment_positives: int = 0):
        """
        Args:
            smiles: List of SMILES strings
            protein_seqs: List of protein sequences
            labels: List of binary labels (0/1)
            augment_positives: Number of SMILES augmentations per positive sample.
                              If > 0, augmented samples are added to the dataset.
        """
        assert len(smiles) == len(protein_seqs) == len(labels)

        self.smiles = list(smiles)
        self.protein_seqs = list(protein_seqs)
        self.labels = list(labels)

        if augment_positives > 0:
            self._augment(augment_positives)

        logging.info(f"  Dataset created: {len(self)} samples "
                     f"(pos={sum(self.labels)}, neg={len(self.labels) - sum(self.labels)})")

    def _augment(self, n_aug: int):
        """Add SMILES augmentations for positive samples."""
        aug_smiles = []
        aug_prots = []
        aug_labels = []
        pos_count = 0
        for s, p, l in zip(self.smiles, self.protein_seqs, self.labels):
            if l == 1:
                augmented = randomize_smiles(s, n_augments=n_aug)
                aug_smiles.extend(augmented)
                aug_prots.extend([p] * len(augmented))
                aug_labels.extend([1] * len(augmented))
                pos_count += 1

        logging.info(f"  Augmented {pos_count} positives with {n_aug} variants each "
                     f"→ {len(aug_smiles)} new samples")
        self.smiles.extend(aug_smiles)
        self.protein_seqs.extend(aug_prots)
        self.labels.extend(aug_labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "smiles": self.smiles[idx],
            "protein_seq": self.protein_seqs[idx],
            "label": self.labels[idx],
        }


class EmbeddedDataset(Dataset):
    """Dataset of pre-computed embeddings + labels for efficient training.

    Optionally carries Morgan fingerprint tensors for hybrid model.
    """

    def __init__(self, protein_embeds: torch.Tensor, ligand_embeds: torch.Tensor,
                 labels: torch.Tensor, morgan_fps: Optional[torch.Tensor] = None):
        assert len(protein_embeds) == len(ligand_embeds) == len(labels)
        if morgan_fps is not None:
            assert len(morgan_fps) == len(labels)
        self.protein_embeds = protein_embeds
        self.ligand_embeds = ligand_embeds
        self.labels = labels
        self.morgan_fps = morgan_fps

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.morgan_fps is not None:
            return (self.protein_embeds[idx], self.ligand_embeds[idx],
                    self.morgan_fps[idx], self.labels[idx])
        return (self.protein_embeds[idx], self.ligand_embeds[idx], self.labels[idx])


def make_dataloader(dataset: EmbeddedDataset, batch_size: int,
                    shuffle: bool = True) -> DataLoader:
    """Create a DataLoader from an EmbeddedDataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, drop_last=False)


def compute_pos_weight(labels: List[int]) -> float:
    """Compute pos_weight for BCEWithLogitsLoss to handle class imbalance."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return 1.0
    weight = n_neg / n_pos
    logging.info(f"  Class balance: pos={n_pos}, neg={n_neg}, pos_weight={weight:.2f}")
    return weight
