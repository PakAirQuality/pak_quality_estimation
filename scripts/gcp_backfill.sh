#!/bin/bash
# =============================================================================
# GCP One-Time Backfill Script
# =============================================================================
#
# Run on a GCP VM with local SSD to backfill the feature store.
#
# Prerequisites:
#   1. VM with local SSD mounted at /mnt/ssd
#   2. Python 3.10+ with dependencies installed
#   3. gcloud authenticated with access to buckets
#
# Usage:
#   ./scripts/gcp_backfill.sh
#
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_ID="${PROJECT_ID:-your-gcp-project}"
REGION="${REGION:-asia-south1}"

# Source buckets (raw data)
RAW_BUCKET="${RAW_BUCKET:-gs://your-raw-bucket}"
MET_PATH="${RAW_BUCKET}/features_met"
AOD_PATH="${RAW_BUCKET}/MCD19A2.061"
TROPOMI_PATH="${RAW_BUCKET}/tropomi_pakistan_2020_2025"
GEOS_CF_PATH="${RAW_BUCKET}/geos_cf_pakistan_2020_2025"
OBSERVATIONS_PATH="${RAW_BUCKET}/paqi_network_daily.csv"

# Destination bucket (derived/processed data)
DERIVED_BUCKET="${DERIVED_BUCKET:-gs://your-derived-bucket}"
STORE_PATH="${DERIVED_BUCKET}/station"

# Local SSD paths (for fast I/O)
SSD_MOUNT="/mnt/ssd"
LOCAL_DATA="${SSD_MOUNT}/data"
LOCAL_MET="${LOCAL_DATA}/features_met"
LOCAL_AOD="${LOCAL_DATA}/MCD19A2.061"
LOCAL_TROPOMI="${LOCAL_DATA}/tropomi_pakistan_2020_2025"
LOCAL_GEOS_CF="${LOCAL_DATA}/geos_cf_pakistan_2020_2025"
LOCAL_OBS="${LOCAL_DATA}/paqi_network_daily.csv"

# Date range for backfill
START_DATE="2020-01-01"
END_DATE="2025-07-01"

# Processing stages
STAGES="met,aod,tropomi"

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_ssd() {
    if [ ! -d "$SSD_MOUNT" ]; then
        log "ERROR: Local SSD not mounted at $SSD_MOUNT"
        log "Run: sudo mkfs.ext4 /dev/nvme0n1 && sudo mount /dev/nvme0n1 $SSD_MOUNT"
        exit 1
    fi
    log "SSD mounted at $SSD_MOUNT"
    df -h "$SSD_MOUNT"
}

create_derived_bucket() {
    log "Checking/creating derived bucket..."
    if ! gsutil ls "$DERIVED_BUCKET" &>/dev/null; then
        log "Creating bucket: $DERIVED_BUCKET"
        gcloud storage buckets create "$DERIVED_BUCKET" \
            --location="$REGION" \
            --project="$PROJECT_ID" \
            --uniform-bucket-level-access
    else
        log "Bucket already exists: $DERIVED_BUCKET"
    fi
}

copy_raw_data() {
    log "Copying raw data from GCS to local SSD..."
    mkdir -p "$LOCAL_DATA"

    # Copy observations CSV
    log "  Copying observations..."
    gsutil -m cp "$OBSERVATIONS_PATH" "$LOCAL_OBS"

    # Copy MET features (NetCDF files)
    log "  Copying MET features..."
    gsutil -m -o "GSUtil:parallel_thread_count=8" rsync -r "$MET_PATH" "$LOCAL_MET"

    # Copy AOD data (HDF files)
    log "  Copying AOD data..."
    gsutil -m -o "GSUtil:parallel_thread_count=8" rsync -r "$AOD_PATH" "$LOCAL_AOD"

    # Copy TROPOMI data (GeoTIFF files)
    log "  Copying TROPOMI data..."
    gsutil -m -o "GSUtil:parallel_thread_count=8" rsync -r "$TROPOMI_PATH" "$LOCAL_TROPOMI"

    # Copy GEOS-CF data
    log "  Copying GEOS-CF data..."
    gsutil -m -o "GSUtil:parallel_thread_count=8" rsync -r "$GEOS_CF_PATH" "$LOCAL_GEOS_CF"

    log "Raw data copy complete."
    du -sh "$LOCAL_DATA"/*
}

run_backfill() {
    log "Starting feature store backfill..."
    log "  Stages: $STAGES"
    log "  Date range: $START_DATE to $END_DATE"
    log "  Output: $STORE_PATH"

    cd "$(dirname "$0")/.."

    python feature_engineering/main_feature_pipeline.py \
        --mode store \
        --stages "$STAGES" \
        --start "$START_DATE" \
        --end "$END_DATE" \
        --daily_csv "$LOCAL_OBS" \
        --features_dir "$LOCAL_MET" \
        --aod_dir "$LOCAL_AOD" \
        --tropomi_base_dir "$LOCAL_TROPOMI" \
        --geos_cf_base_dir "$LOCAL_GEOS_CF" \
        --store_path "$STORE_PATH" \
        --output_parquet "${LOCAL_DATA}/master.parquet" \
        --verbose

    log "Backfill complete."
}

upload_master() {
    log "Uploading master.parquet to GCS..."
    gsutil cp "${LOCAL_DATA}/master.parquet" "${DERIVED_BUCKET}/master.parquet"
    log "Master dataset uploaded: ${DERIVED_BUCKET}/master.parquet"
}

cleanup() {
    log "Cleaning up local SSD..."
    # Uncomment to delete local data after successful upload
    # rm -rf "$LOCAL_DATA"
    log "Cleanup complete (local data preserved for debugging)"
}

show_summary() {
    log "============================================================"
    log "BACKFILL COMPLETE"
    log "============================================================"
    log "Feature store: $STORE_PATH"
    log "Master dataset: ${DERIVED_BUCKET}/master.parquet"
    log ""
    log "To verify:"
    log "  gsutil ls ${STORE_PATH}/met/ | head"
    log "  gsutil ls ${STORE_PATH}/aod/ | head"
    log "  gsutil ls ${STORE_PATH}/tropomi/ | head"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    log "============================================================"
    log "GCP FEATURE STORE BACKFILL"
    log "============================================================"

    # Step 1: Check SSD is mounted
    check_ssd

    # Step 2: Create derived bucket if needed
    create_derived_bucket

    # Step 3: Copy raw data to local SSD
    copy_raw_data

    # Step 4: Run backfill pipeline
    run_backfill

    # Step 5: Upload master dataset
    upload_master

    # Step 6: Cleanup
    cleanup

    # Step 7: Summary
    show_summary
}

# Run main if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
