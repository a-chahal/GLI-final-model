"""
GLI-PLAPT Model — Baseline PLAPT (ProtBERT) and Modified PLAPT (ESM-2) architectures.

Both models share:
    - Frozen pretrained encoders (protein + ligand)
    - Branching prediction head with MC Dropout
    - Binary classification output
"""

import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel,
    EsmTokenizer, EsmModel,
)

from src.config import Config, PredictionHeadConfig


# ---------------------------------------------------------------------------
# Prediction Head (shared by both baseline and modified)
# ---------------------------------------------------------------------------

class BranchingPredictionHead(nn.Module):
    """Branching neural network with MC Dropout for binary classification.

    Architecture:
        protein_emb → Linear → ReLU → Dropout ─┐
                                                 ├─ concat + hadamard → Fusion MLP → logit
        ligand_emb  → Linear → ReLU → Dropout ─┘

    The Hadamard product (element-wise p⊙l) provides explicit
    protein-ligand interaction features alongside the concatenation.

    At inference with MC Dropout, run T forward passes with dropout enabled
    to obtain mean prediction and uncertainty estimate.
    """

    def __init__(self, protein_dim: int, ligand_dim: int, cfg: PredictionHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.use_morgan_fp = cfg.use_morgan_fp

        # Protein branch
        self.protein_branch = nn.Sequential(
            nn.Linear(protein_dim, cfg.protein_branch_out),
            nn.ReLU(),
            nn.Dropout(cfg.branch_dropout),
        )

        # Ligand branch
        self.ligand_branch = nn.Sequential(
            nn.Linear(ligand_dim, cfg.ligand_branch_out),
            nn.ReLU(),
            nn.Dropout(cfg.branch_dropout),
        )

        # Optional Morgan fingerprint branch (Phase 2C)
        morgan_out = 0
        if self.use_morgan_fp:
            self.morgan_branch = nn.Sequential(
                nn.Linear(cfg.morgan_fp_bits, cfg.morgan_fp_hidden),
                nn.ReLU(),
                nn.Dropout(cfg.branch_dropout),
            )
            morgan_out = cfg.morgan_fp_hidden

        # Fusion layers: concat + hadamard product for interaction features
        # Input = [prot_branch; lig_branch; prot_branch ⊙ lig_branch; morgan_branch?]
        fusion_in = cfg.protein_branch_out + cfg.ligand_branch_out + cfg.protein_branch_out + morgan_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, cfg.fusion_hidden_1),
            nn.ReLU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(cfg.fusion_hidden_1, cfg.fusion_hidden_2),
            nn.ReLU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(cfg.fusion_hidden_2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, protein_emb: torch.Tensor, ligand_emb: torch.Tensor,
                morgan_fp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass returning raw logit (no sigmoid).

        Args:
            protein_emb: (batch, protein_dim)
            ligand_emb: (batch, ligand_dim)
            morgan_fp: (batch, morgan_fp_bits) optional Morgan fingerprints

        Returns:
            logit: (batch, 1)
        """
        p = self.protein_branch(protein_emb)
        l = self.ligand_branch(ligand_emb)
        interaction = p * l  # Hadamard product for explicit interaction features
        parts = [p, l, interaction]

        if self.use_morgan_fp:
            if morgan_fp is not None:
                m = self.morgan_branch(morgan_fp)
            else:
                # No Morgan FPs provided (Stage 1/2) — pass zeros to keep dims consistent
                m = self.morgan_branch(
                    torch.zeros(protein_emb.shape[0], self.cfg.morgan_fp_bits,
                                device=protein_emb.device)
                )
            parts.append(m)

        fused = torch.cat(parts, dim=1)
        return self.fusion(fused)

    def mc_predict(self, protein_emb: torch.Tensor, ligand_emb: torch.Tensor,
                   n_samples: Optional[int] = None,
                   morgan_fp: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """MC Dropout inference for uncertainty estimation.

        Runs T stochastic forward passes with dropout enabled.

        Returns:
            dict with keys:
                mean_prob: (batch,) mean predicted probability
                std_prob: (batch,) standard deviation (uncertainty)
                all_probs: (batch, T) all sampled probabilities
        """
        T = n_samples or self.cfg.mc_samples
        self.train()  # Enable dropout

        logits = []
        with torch.no_grad():
            for _ in range(T):
                logit = self.forward(protein_emb, ligand_emb, morgan_fp=morgan_fp)
                logits.append(logit.squeeze(-1))

        self.eval()
        logits = torch.stack(logits, dim=1)  # (batch, T)
        probs = torch.sigmoid(logits)        # (batch, T)

        return {
            "mean_prob": probs.mean(dim=1),
            "std_prob": probs.std(dim=1),
            "all_probs": probs,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Encoder Wrapper
# ---------------------------------------------------------------------------

class EncoderWrapper:
    """Wraps frozen pretrained encoders for embedding computation.

    This is NOT an nn.Module — it's a utility for pre-computing embeddings.
    Encoders are always frozen and used in eval mode with no_grad.
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.use_esm2 = config.use_esm2

        logging.info("Loading encoders...")

        # Protein encoder
        if self.use_esm2:
            logging.info(f"  Protein: ESM-2 ({config.encoder.esm2_model_name})")
            self.prot_tokenizer = EsmTokenizer.from_pretrained(config.encoder.esm2_model_name)
            self.prot_encoder = EsmModel.from_pretrained(config.encoder.esm2_model_name)
            self.prot_dim = config.encoder.esm2_embed_dim
            self.prot_max_len = config.encoder.esm2_max_length
            self.prot_model_name = config.encoder.esm2_model_name
        else:
            logging.info(f"  Protein: ProtBERT ({config.encoder.protbert_model_name})")
            self.prot_tokenizer = BertTokenizer.from_pretrained(
                config.encoder.protbert_model_name, do_lower_case=False
            )
            self.prot_encoder = BertModel.from_pretrained(config.encoder.protbert_model_name)
            self.prot_dim = config.encoder.protbert_embed_dim
            self.prot_max_len = config.encoder.protbert_max_length
            self.prot_model_name = config.encoder.protbert_model_name

        # Ligand encoder (always ChemBERTa)
        logging.info(f"  Ligand: ChemBERTa ({config.encoder.chemberta_model_name})")
        self.mol_tokenizer = RobertaTokenizer.from_pretrained(config.encoder.chemberta_model_name)
        self.mol_encoder = RobertaModel.from_pretrained(config.encoder.chemberta_model_name)
        self.mol_dim = config.encoder.chemberta_embed_dim
        self.mol_model_name = config.encoder.chemberta_model_name

        # Freeze and move to device
        self.prot_encoder.eval()
        self.mol_encoder.eval()
        for param in self.prot_encoder.parameters():
            param.requires_grad = False
        for param in self.mol_encoder.parameters():
            param.requires_grad = False

        self.prot_encoder.to(device)
        self.mol_encoder.to(device)
        logging.info("  Encoders loaded and frozen.")

    def encode_protein(self, sequence: str) -> torch.Tensor:
        """Encode a single protein sequence → (embed_dim,) tensor."""
        if not self.use_esm2:
            # ProtBERT requires space-separated sequence
            import re
            sequence = " ".join(re.sub(r"[UZOB]", "X", sequence))

        tokens = self.prot_tokenizer(
            sequence,
            padding=True,
            truncation=True,
            max_length=self.prot_max_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.prot_encoder(**tokens)
        # Mean pool over sequence length (excluding special tokens)
        # For ESM-2, last_hidden_state is better than pooler_output
        # For ProtBERT, pooler_output is standard (but mean pool is also fine for consistency)
        emb = output.last_hidden_state[:, 1:-1, :].mean(dim=1)  # skip [CLS] and [EOS]
        return emb.squeeze(0).cpu()

    def encode_ligand(self, smiles: str) -> torch.Tensor:
        """Encode a single SMILES string → (embed_dim,) tensor."""
        tokens = self.mol_tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=self.config.encoder.chemberta_max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.mol_encoder(**tokens)
        # Use pooler_output for ChemBERTa (matches original PLAPT)
        return output.pooler_output.squeeze(0).cpu()

    def encode_proteins_batch(self, sequences: List[str], batch_size: int = 2) -> torch.Tensor:
        """Encode a list of protein sequences with batching."""
        all_embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            if not self.use_esm2:
                import re
                batch = [" ".join(re.sub(r"[UZOB]", "X", s)) for s in batch]

            tokens = self.prot_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.prot_max_len,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self.prot_encoder(**tokens)
            emb = output.last_hidden_state[:, 1:-1, :].mean(dim=1)
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode_ligands_batch(self, smiles_list: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode a list of SMILES with batching."""
        all_embeddings = []
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            tokens = self.mol_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.encoder.chemberta_max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self.mol_encoder(**tokens)
            all_embeddings.append(output.pooler_output.cpu())

        return torch.cat(all_embeddings, dim=0)

    def offload(self):
        """Move encoders off GPU to free memory for training."""
        self.prot_encoder.cpu()
        self.mol_encoder.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("  Encoders offloaded from GPU.")
