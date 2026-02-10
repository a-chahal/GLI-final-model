#!/bin/bash

# Configuration
PROJECT_DIR=$(pwd)
CONDA_ENV_NAME="gli-plapt"
DATA_DIR="$PROJECT_DIR/data"
OUTPUT_DIR="$PROJECT_DIR/outputs"
LOG_DIR="$OUTPUT_DIR/logs"

# Ensure output directories exist
mkdir -p "$DATA_DIR/collected"
mkdir -p "$LOG_DIR"

echo "=============================================================================="
echo "STARTING GLI INHIBITOR DISCOVERY PIPELINE ON ROSENBLUTH"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Project Dir: $PROJECT_DIR"
echo "Log Dir: $LOG_DIR"
echo "=============================================================================="

# Activate Conda Environment
echo "Activating conda environment: $CONDA_ENV_NAME..."
# Initialize conda for shell script
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

if [ $? -ne 0 ]; then
    echo "ERROR: Could not activate conda environment '$CONDA_ENV_NAME'"
    exit 1
fi

# 1. Collect Compounds
echo ""
echo "[1/3] Collecting compounds from databases (ChEMBL, PubChem, ZINC, DGIdb)..."
python -m src.collect_compounds \
    --output-dir "$DATA_DIR/collected" \
    --sources chembl pubchem zinc dgidb \
    --skip-existing \
    2>&1 | tee "$LOG_DIR/collect_compounds.log"

if [ $? -ne 0 ]; then
    echo "ERROR: Compound collection failed. Check $LOG_DIR/collect_compounds.log"
    exit 1
fi
echo "Collection complete. Data saved to $DATA_DIR/collected"

# 2. Pre-screen Filter (100 CPU cores)
echo ""
echo "[2/3] Running pre-screening filter cascade (CPU Optimized)..."
# Use 100 workers for the 120-core CPU
python -m src.prescreen_filter \
    --input "$DATA_DIR/collected/all_collected_compounds.csv" \
    --output "$DATA_DIR/prescreened_compounds.csv" \
    --reference "$PROJECT_DIR/gli_inhibitors.csv" \
    --workers 100 \
    --tanimoto-threshold 0.3 \
    2>&1 | tee "$LOG_DIR/prescreen_filter.log"

if [ $? -ne 0 ]; then
    echo "ERROR: Pre-screening failed. Check $LOG_DIR/prescreen_filter.log"
    exit 1
fi
echo "Pre-screening complete. Data saved to $DATA_DIR/prescreened_compounds.csv"

# 3. Virtual Screening (4x 3090 GPUs + 100 CPU workers)
echo ""
echo "[3/3] Running AI virtual screening (Multi-GPU Inference)..."
# Use 4 GPUs for encoding and 100 CPU workers for data loading
python -m src.screen_compounds \
    --input "$DATA_DIR/prescreened_compounds.csv" \
    --output "$OUTPUT_DIR/final_screening_results.csv" \
    --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
    --gpus 4 \
    --workers 100 \
    --batch-size 1024 \
    --mc-samples 50 \
    2>&1 | tee "$LOG_DIR/screen_compounds.log"

if [ $? -ne 0 ]; then
    echo "ERROR: Virtual screening failed. Check $LOG_DIR/screen_compounds.log"
    exit 1
fi

echo ""
echo "=============================================================================="
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "Final results: $OUTPUT_DIR/final_screening_results.csv"
echo "=============================================================================="
