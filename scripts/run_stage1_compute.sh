#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
REPO_DIR="/home/vatsal1_hawk_franklin_research_c/pelliscope-train"
VENV_DIR="/home/vatsal1_hawk_franklin_research_c/.venv"

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

mkdir -p reports/production_logs
LOG_FILE="reports/production_logs/stage1_compute.log"

echo "=================================================================" | tee -a "$LOG_FILE"
echo "  PELLISCOPE 25-CLASS FULL REPRODUCTION PIPELINE (STAGE 1 COMPUTE)" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"
echo "Start Time:   $(date -u +"%Y-%m-%d %H:%M:%S UTC")" | tee -a "$LOG_FILE"
echo "Host:         $(hostname)" | tee -a "$LOG_FILE"
echo "CPUs:         $(nproc)" | tee -a "$LOG_FILE"
echo "Memory:       $(free -h | awk '/^Mem:/ {print $2}') total, $(free -h | awk '/^Mem:/ {print $7}') available" | tee -a "$LOG_FILE"
echo "Disk:         $(df -h "$REPO_DIR" | awk 'NR==2 {print $4}') free" | tee -a "$LOG_FILE"
echo "Git Branch:   $(git rev-parse --abbrev-ref HEAD)" | tee -a "$LOG_FILE"
echo "Git Commit:   $(git rev-parse HEAD)" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

# Clean any previous flags
rm -f reports/production_logs/STAGE1_SUCCESS.flag reports/production_logs/STAGE1_FAILED.flag

# Run pipeline
if python3 scripts/run_pipeline.py \
    --config configs/study_25class.yaml \
    --run-mode full \
    --primary-encoder siglip2_so400m \
    --device cpu 2>&1 | tee -a "$LOG_FILE"; then

    echo "" | tee -a "$LOG_FILE"
    echo "Pipeline process completed with exit code 0. Verifying output integrity..." | tee -a "$LOG_FILE"

    # Explicit post-run release verification
    python3 scripts/70_verify_release.py \
        --config configs/study_25class.yaml \
        --encoders resnet50,inception_v3,bit50,vit_base,clip_vitb32,derm_foundation,siglip2_so400m \
        --primary-encoder siglip2_so400m 2>&1 | tee -a "$LOG_FILE"

    # Integrity Gate: Verify critical artifacts are non-empty and non-null
    MISSING_COUNT=0
    for encoder in resnet50 inception_v3 bit50 vit_base clip_vitb32 derm_foundation siglip2_so400m; do
        BANK="artifacts/features/$encoder/feature_bank.npz"
        if [ ! -s "$BANK" ]; then
            echo "ERROR: Missing or empty feature bank: $BANK" | tee -a "$LOG_FILE"
            MISSING_COUNT=$((MISSING_COUNT + 1))
        fi
    done

    MODEL="artifacts/models/mil/siglip2_so400m/final_model/model.pt"
    if [ ! -s "$MODEL" ]; then
        echo "ERROR: Missing or empty final model: $MODEL" | tee -a "$LOG_FILE"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi

    PREDS="artifacts/models/mil/siglip2_so400m/final_model/test_predictions.csv"
    if [ ! -s "$PREDS" ]; then
        echo "ERROR: Missing or empty test predictions: $PREDS" | tee -a "$LOG_FILE"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi

    TABLES="reports/tables/all_model_metrics.csv"
    if [ ! -s "$TABLES" ]; then
        echo "ERROR: Missing or empty summary tables: $TABLES" | tee -a "$LOG_FILE"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi

    if [ "$MISSING_COUNT" -eq 0 ]; then
        echo "All integrity checks passed! Flushing disk..." | tee -a "$LOG_FILE"
        sync
        touch reports/production_logs/STAGE1_SUCCESS.flag
        echo "STAGE 1 SUCCESS recorded at $(date -u +"%Y-%m-%d %H:%M:%S UTC")" | tee -a "$LOG_FILE"
    else
        echo "Integrity check failed with $MISSING_COUNT missing/empty artifacts." | tee -a "$LOG_FILE"
        touch reports/production_logs/STAGE1_FAILED.flag
        exit 1
    fi
else
    echo "Pipeline exited with non-zero status code." | tee -a "$LOG_FILE"
    touch reports/production_logs/STAGE1_FAILED.flag
    exit 1
fi
