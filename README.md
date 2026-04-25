# GLI-NT

**AI-Guided Discovery of Novel GLI1 Inhibitors for Medulloblastoma**

A multi-modal deep learning framework for identifying small-molecule inhibitors of the GLI1 zinc finger–DNA interface, the terminal transcriptional node of the Sonic Hedgehog pathway. GLI-NT combines cascaded transfer learning across three data regimes with docking- and molecular dynamics–based biophysical validation.

*Greater San Diego Science and Engineering Fair, Senior Division. Aaran Chahal and Sanjan Prabu.*

---

## Why GLI1

Medulloblastoma is the most common malignant pediatric brain tumor; ~30% of cases are driven by aberrant Hedgehog signaling. Approved inhibitors target Smoothened (SMO) upstream of GLI and lose efficacy within months as tumors acquire resistance mutations or bypass SMO through PI3K/AKT and RAS/MAPK. GLI1 sits at the terminal node — blocking it circumvents every known resistance mechanism. No direct GLI1 inhibitor exists clinically. The zinc finger–DNA interface is shallow and solvent-exposed, which defeats conventional structure-based screening. GLI-NT takes a different path: learn binding from data, validate with biophysics.

---

## Method Overview

GLI-NT has four components:

1. **Multi-modal encoder stack (frozen).** ESM-2 (`esm2_t33_650M_UR50D`, 1280-dim per residue → mean-pooled) for protein sequence; ChemBERTa (`DeepChem/ChemBERTa-77M-MLM`, 384-dim `[CLS]`) for SMILES chemistry; 2,048-bit Morgan (ECFP4) fingerprints for explicit substructure. All three encoders are frozen — only the prediction head is trainable.
2. **Trainable prediction head.** A `BranchingPredictionHead` that maps each modality through its own MLP branch, then fuses via concatenation plus a protein–ligand Hadamard product, followed by a two-layer MLP with dropout. Monte Carlo Dropout at inference yields per-prediction uncertainty. No GNN is used.
3. **Cascaded transfer learning (three stages).**
   - *Stage 1 — Pretraining:* 150,935 BindingDB pairs with random protein-ligand shuffling as negatives.
   - *Stage 2 — Domain adaptation:* 1,573 zinc-finger binding pairs with a 1 µM active/inactive threshold.
   - *Stage 3 — GLI-specific fine-tuning:* leave-one-out cross-validation (LOOCV) across 28 known GLI binders, with focal loss, asymmetric SMILES augmentation, and Youden's J threshold calibration per fold. Repeated across five random seeds.
4. **Biophysical validation pipeline.** Candidates pass through PAINS filtering, Butina clustering for scaffold diversity (Tanimoto cutoff 0.35), dual-site AutoDock Vina docking to the 2GLI crystal structure (ZF2–3 and ZF4–5 sites, exhaustiveness 32, 10 poses, 22 Å cubic box), and ternary molecular dynamics with GROMACS (production target 200 ns; ~110 ns analyzed per validated candidate; AMBER99SB-ILDN + bsc0 for protein/DNA, GAFF2/AM1-BCC for ligand, ZAFF bonded zinc model, TIP3P water, 150 mM NaCl, Parrinello-Rahman barostat).

---

## Repository Structure

```
.
├── src/                           # Core training pipeline
│   ├── config.py                  # All hyperparameters, seeds, paths
│   ├── data.py                    # Data loading, Morgan FP, augmentation, MD5-hashed embedding cache
│   ├── model.py                   # EncoderWrapper + BranchingPredictionHead
│   ├── trainer.py                 # Training loop, focal loss, Youden's J threshold
│   ├── evaluate.py                # LOOCV orchestration + statistical tests
│   ├── analysis.py                # Tanimoto analysis, consensus, multi-seed aggregation
│   └── utils.py                   # Seeding, logging, checkpointing
│
├── run.py                         # Main entry point: Stage 1 → 2 → 3 pipeline
├── run_ablation.py                # Six-condition ablation study orchestrator
├── screen_compounds.py            # Multi-GPU virtual screening with MC Dropout ensemble
├── extract_hits.py                # Hit extraction, novelty analysis, Butina clustering
├── light_prescreen.py             # Lightweight prescreening utilities
├── prescreen_filter.py            # Drug-like + PAINS filtering
│
├── batch_dock_dual_sites.py       # AutoDock Vina docking to ZF2–3 and ZF4–5
├── prepare_receptor_validated.py  # 2GLI receptor preparation
├── fix_receptor_zinc.py           # Zn²⁺ coordination fix for Vina
├── visualize_docking_sites.py     # Binding-site visualization
├── visualize_top_hits.py          # Per-hit docking pose visualization
│
├── md_simulation/                 # Binary MD pipeline (protein + ligand)
│   ├── run_md_pipeline.py
│   ├── dock_and_run_references.py
│   ├── plot_md_results.py
│   └── mdp/                       # GROMACS .mdp files (em, nvt, npt, production)
│
├── ternary_md/                    # Ternary MD pipeline (protein + DNA + ligand)
│   └── run_ternary_pipeline.py    # GAFF2 ligand prep, ZAFF zinc, AMBER99SB-ILDN (production target 200 ns; ~110 ns analyzed per validated candidate)
│
├── generate_figures.py            # Publication-quality figure generation
├── generate_notebook_figures.py
├── analyze_expanded.py            # Post-hoc analysis
├── analyze_novelty.py             # Scaffold novelty auditing
├── collect_compounds.py           # Compound aggregation from multiple sources
│
├── 2gli_*.pdb / .pdbqt            # Public GLI1 crystal structure (PDB: 2GLI)
├── gli1_sequence.fasta            # GLI1 zinc finger domain sequence
├── example_compounds.csv          # Example input (GANT61, CHEMBL16958)
├── environment.yml                # Conda environment
└── requirements.txt               # Pip requirements
```

**Not in this repository (intentionally):** training datasets, trained checkpoints, screening outputs, docking poses for identified candidates, MD trajectories, and the detailed compound tier tables. Candidate compound identities are withheld pending further work. All scripts run end-to-end once the data paths in `src/config.py` are populated.

---

## Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate gli-nt

# Additional pip-only dependencies
pip install -r requirements.txt
```

Key dependencies: PyTorch 2.1 (CUDA 11.8), `transformers>=4.35`, `rdkit`, `scikit-learn`, `biopython`, AutoDock Vina Python bindings, and GROMACS 2023+ (for MD).

---

## Reproducing the Training Pipeline

Data sources (not redistributed here; see links):

| Dataset | Role | Source |
|---|---|---|
| BindingDB | Stage 1 pretraining | https://www.bindingdb.org/ |
| UniProt | Protein sequences | https://www.uniprot.org/ |
| ChEMBL | Zinc finger + GLI supplementary | https://www.ebi.ac.uk/chembl/ |
| GLI1 known inhibitors (n=28) | Stage 3 LOOCV positives | Compiled from published literature; see notebook §4.4 |
| Enamine REAL / COCONUT | Virtual screening libraries | https://enamine.net, https://coconut.naturalproducts.net |

Once data files are placed per `src/config.py`:

```bash
# Full pipeline: embedding precomputation → Stage 1 → Stage 2 → Stage 3 LOOCV
python run.py

# Ablation study (6 conditions)
python run_ablation.py

# Virtual screening (multi-GPU)
python screen_compounds.py --library <path>.csv --output-dir outputs/

# Hit extraction + Butina diversity selection
python extract_hits.py --input outputs/screening_results.csv --top-k 500

# Docking to both GLI1 binding sites
python batch_dock_dual_sites.py hits.csv --exhaustiveness 32

# Ternary MD (200 ns, protein + DNA + ligand + Zn²⁺)
python ternary_md/run_ternary_pipeline.py
```

---

## Key Implementation Details

- **Embedding cache with MD5-hashed keys** (`src/data.py:EmbeddingCache`). `md5(f"{model_name}:{text}")` → `.pt` file. With frozen encoders, the same `(sequence, model)` pair always maps to the same cached tensor, giving exact cross-run reproducibility regardless of seed or ablation condition.
- **Per-fold Youden's J threshold calibration** (`src/trainer.py:find_optimal_threshold`). Each LOOCV fold computes its own optimal threshold on a 10% within-fold validation split by maximizing sensitivity + specificity − 1 over a grid from 0.05 to 0.95.
- **Monte Carlo Dropout at inference** (`src/model.py:BranchingPredictionHead.mc_predict`). 30 forward passes with dropout active; the standard deviation of the resulting probability distribution is the per-prediction uncertainty.
- **Focal loss in Stage 3** (`src/trainer.py:FocalLoss`). γ=2.0, α=0.75. Down-weights easy analogs and focuses gradient on structurally diverse hard positives.
- **Butina clustering for scaffold diversity** (`extract_hits.py:butina_cluster`). Clusters top predictions by Tanimoto distance at a 0.35 cutoff; one representative per cluster is selected, ensuring the final set is chemically diverse rather than a narrow analog series.
- **Novelty scoring** (`extract_hits.py:novelty_analysis`). Each hit's maximum Tanimoto similarity to the 28 known GLI binders is computed. Compounds with max Tc < 0.4 are flagged as novel scaffolds.
- **Five-seed replication.** All LOOCV runs repeat across seeds 42, 123, 456, 789, 1024 (`src/config.py:MULTI_SEEDS`).

---

## Headline Results

All numbers are from the five-seed LOOCV over 28 known GLI binders. Full numerical breakdown and per-compound tables are in the notebook (not included in this repository).

- **LOOCV hit rate:** 72.1 ± 4.7% across five seeds (n = 28 known GLI binders). Three compounds were consistently missed across all seeds, each with Tanimoto < 0.25 to any training-set neighbor — confirming scaffold-limited generalization as the true ceiling, not model capacity.
- **Within-fold validation AUROC:** very high (~0.99). This measures separation of known binders from non-binders on the within-fold 10% split, *not* held-out-compound prediction; it reflects strong feature quality from the frozen pretrained encoders and a chemically distinct negative set.
- **Ablation:** removing either the multi-modal fusion, the cascaded pretraining, or the focal loss each causes a measurable drop in held-out hit rate. Details in the notebook.
- **Virtual screening:** >1 M candidate compounds scored → filtered to a diverse set of 500 via PAINS + Butina clustering; 84.6% of final candidates have max Tc < 0.4 to any known GLI binder (novel scaffolds).
- **Docking:** best binding energies −8.70 kcal/mol at ZF2-3 and −8.52 kcal/mol at ZF4-5 (AutoDock Vina, 10 poses per run).
- **Ternary MD validation:** 200-ns simulations with protein + DNA + ligand + Zn²⁺ distinguished sustained target engagement from apparent-binders that dissociated when DNA competed for the same surface, and from intercalators that bound DNA rather than the protein. These categorical outcomes are the primary biophysical signal — absolute numbers per compound are withheld.

---

## What This Project Does *Not* Do

To prevent overclaim:

- **No wet-lab validation yet.** All results are computational. The claim is hypothesis generation, not pharmacological efficacy.
- **No ADMET / PK modeling beyond standard Lipinski/QED and PAINS filters.**
- **No GNN or graph transformer.** Ligand chemistry is learned from SMILES via ChemBERTa, plus explicit Morgan fingerprints for substructure. Anyone comparing this to GNN-based pipelines should note the architecture is MLP-fusion over pretrained transformer embeddings.
- **AUROC is not the success metric.** The within-fold AUROC is ~0.99 but reflects interpolation within the training pool. The meaningful metric is the LOOCV held-out hit rate.
- **Candidate compound IDs are withheld** from this public repository pending further work.

---

## Computational Resources

Model training and virtual screening were performed on the Rosenbluth workstation at UC San Diego (4× NVIDIA RTX 3090, 2× 60-core CPU), made available by Dr. Michael K. Gilson (Skaggs School of Pharmacy and Pharmaceutical Sciences, UCSD). Ternary MD simulations used GROMACS 2023 with GPU acceleration. Dr. Leenus Martin and Dr. Kirti Kandhwal Chahal provided mentorship on docking methodology and medicinal chemistry context.

---

## Citation

If this work or codebase is useful to you, please cite:

```
Chahal, A. and Prabu, S. (2026). GLI-NT: AI-Guided Discovery of Novel GLI1
Inhibitors for Medulloblastoma. Greater San Diego Science and Engineering Fair,
Senior Division.
```

---

## License and Contact

Code is released for non-commercial research and educational use. For commercial licensing, collaboration, or questions, contact the authors through the repository's issues page.
