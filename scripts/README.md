# GCP Deployment Scripts

```
scripts/
├── setup_vm.sh       # VM first-time setup
└── gcp_backfill.sh   # Full backfill workflow
```

## Quick Start on GCP

### 1. Create VM (with local SSD)

```bash
gcloud compute instances create paqi-backfill \
  --zone=asia-south1-a \
  --machine-type=n2-standard-8 \
  --local-ssd=interface=NVME \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud
```

### 2. SSH and run setup

```bash
gcloud compute ssh paqi-backfill --zone=asia-south1-a

# Clone repo and run setup
git clone YOUR_REPO ~/Estimation
cd ~/Estimation
./scripts/setup_vm.sh
```

### 3. Run backfill (use tmux for long jobs)

```bash
tmux new -s backfill
./scripts/gcp_backfill.sh
# Ctrl+B, D to detach
```

## What the Backfill Does

```
1. Copy raw data → /mnt/ssd/data/
   ├── features_met/
   ├── MCD19A2.061/
   ├── tropomi_pakistan_2020_2025/
   └── paqi_network_daily.csv

2. Process (met, aod, tropomi) → Write to GCS
   gs://your-derived-bucket/station/
   ├── met/date=2020-01-01/part-000.parquet
   ├── aod/date=2020-01-01/part-000.parquet
   └── tropomi/date=2020-01-01/part-000.parquet

3. Build & upload master.parquet
```

## Daily Updates (after backfill)

After the one-time backfill, daily updates only process new dates:

```bash
python feature_engineering/main_feature_pipeline.py \
  --mode store \
  --stages met,aod,tropomi \
  --start $(date -d "yesterday" +%Y-%m-%d) \
  --end $(date -d "yesterday" +%Y-%m-%d) \
  --store_path gs://your-derived-bucket/station
```

The pipeline automatically skips existing partitions (incremental processing).
