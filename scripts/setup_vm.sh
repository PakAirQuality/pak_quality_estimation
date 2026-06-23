#!/bin/bash
# =============================================================================
# GCP VM Setup Script
# =============================================================================
#
# Sets up a fresh GCP VM for running the feature engineering pipeline.
#
# VM Requirements:
#   - Machine type: n2-standard-8 or larger (8 vCPUs, 32GB RAM)
#   - Local SSD: 375GB NVMe (for fast I/O)
#   - OS: Ubuntu 22.04 LTS
#   - Region: asia-south1 (same as data buckets)
#
# Usage:
#   # SSH into the VM, then:
#   curl -O https://raw.githubusercontent.com/YOUR_REPO/main/scripts/setup_vm.sh
#   chmod +x setup_vm.sh
#   ./setup_vm.sh
#
# =============================================================================

set -e

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# -----------------------------------------------------------------------------
# System Setup
# -----------------------------------------------------------------------------

setup_system() {
    log "Updating system packages..."
    sudo apt-get update
    sudo apt-get upgrade -y

    log "Installing system dependencies..."
    sudo apt-get install -y \
        build-essential \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        libhdf4-dev \
        libhdf5-dev \
        libnetcdf-dev \
        libgdal-dev \
        gdal-bin \
        git \
        htop \
        tmux
}

# -----------------------------------------------------------------------------
# Mount Local SSD
# -----------------------------------------------------------------------------

mount_ssd() {
    log "Setting up local SSD..."

    SSD_DEVICE="/dev/nvme0n1"
    SSD_MOUNT="/mnt/ssd"

    # Check if SSD exists
    if [ ! -b "$SSD_DEVICE" ]; then
        log "WARNING: Local SSD not found at $SSD_DEVICE"
        log "Make sure VM was created with a local SSD attached"
        return 1
    fi

    # Format if needed (check if already formatted)
    if ! blkid "$SSD_DEVICE" &>/dev/null; then
        log "Formatting SSD..."
        sudo mkfs.ext4 -F "$SSD_DEVICE"
    fi

    # Create mount point
    sudo mkdir -p "$SSD_MOUNT"

    # Mount
    sudo mount "$SSD_DEVICE" "$SSD_MOUNT"

    # Set permissions
    sudo chown -R "$USER:$USER" "$SSD_MOUNT"

    # Add to fstab for persistence
    if ! grep -q "$SSD_DEVICE" /etc/fstab; then
        echo "$SSD_DEVICE $SSD_MOUNT ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
    fi

    log "SSD mounted at $SSD_MOUNT"
    df -h "$SSD_MOUNT"
}

# -----------------------------------------------------------------------------
# Python Environment
# -----------------------------------------------------------------------------

setup_python() {
    log "Setting up Python environment..."

    VENV_DIR="$HOME/venv"

    # Create virtual environment
    python3.10 -m venv "$VENV_DIR"

    # Activate
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip wheel setuptools

    log "Python environment ready at $VENV_DIR"
}

install_dependencies() {
    log "Installing Python dependencies..."

    source "$HOME/venv/bin/activate"

    # Install from requirements.txt if available
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        # Install core dependencies
        pip install \
            numpy>=1.24.0 \
            pandas>=2.0.0 \
            scipy>=1.10.0 \
            tqdm>=4.65.0 \
            xarray>=2023.1.0 \
            netCDF4>=1.6.0 \
            h5py>=3.8.0 \
            pyhdf>=0.10.5 \
            rasterio>=1.3.0 \
            pyarrow>=14.0.0 \
            gcsfs>=2023.1.0
    fi

    log "Dependencies installed."
}

# -----------------------------------------------------------------------------
# Clone Repository
# -----------------------------------------------------------------------------

clone_repo() {
    log "Cloning repository..."

    REPO_URL="https://github.com/YOUR_ORG/Estimation.git"  # Update this
    REPO_DIR="$HOME/Estimation"

    if [ -d "$REPO_DIR" ]; then
        log "Repository already exists, pulling latest..."
        cd "$REPO_DIR"
        git pull
    else
        git clone "$REPO_URL" "$REPO_DIR"
    fi

    cd "$REPO_DIR"
    log "Repository ready at $REPO_DIR"
}

# -----------------------------------------------------------------------------
# Verify Setup
# -----------------------------------------------------------------------------

verify_setup() {
    log "Verifying setup..."

    source "$HOME/venv/bin/activate"

    # Check Python
    python --version

    # Check key imports
    python -c "import pandas; print(f'pandas: {pandas.__version__}')"
    python -c "import xarray; print(f'xarray: {xarray.__version__}')"
    python -c "import rasterio; print(f'rasterio: {rasterio.__version__}')"
    python -c "import gcsfs; print(f'gcsfs: {gcsfs.__version__}')"

    # Check SSD
    df -h /mnt/ssd

    # Check gcloud
    gcloud auth list

    log "Setup verification complete!"
}

# -----------------------------------------------------------------------------
# Create convenience aliases
# -----------------------------------------------------------------------------

setup_aliases() {
    log "Setting up aliases..."

    cat >> "$HOME/.bashrc" << 'EOF'

# Feature engineering pipeline aliases
alias activate='source ~/venv/bin/activate'
alias backfill='cd ~/Estimation && ./scripts/gcp_backfill.sh'
alias logs='tail -f ~/Estimation/feature_engineering/logs/*.log'

# Activate venv on login
source ~/venv/bin/activate
EOF

    log "Aliases added. Run 'source ~/.bashrc' to apply."
}

# -----------------------------------------------------------------------------
# Print next steps
# -----------------------------------------------------------------------------

print_next_steps() {
    log "============================================================"
    log "SETUP COMPLETE"
    log "============================================================"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Activate the Python environment:"
    echo "   source ~/venv/bin/activate"
    echo ""
    echo "2. Update the repository URL in setup_vm.sh and clone:"
    echo "   git clone YOUR_REPO ~/Estimation"
    echo ""
    echo "3. Install dependencies:"
    echo "   cd ~/Estimation && pip install -r requirements.txt"
    echo ""
    echo "4. Run the backfill:"
    echo "   ./scripts/gcp_backfill.sh"
    echo ""
    echo "5. (Optional) Use tmux for long-running jobs:"
    echo "   tmux new -s backfill"
    echo "   ./scripts/gcp_backfill.sh"
    echo "   # Ctrl+B, D to detach"
    echo ""
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    log "============================================================"
    log "GCP VM SETUP"
    log "============================================================"

    setup_system
    mount_ssd
    setup_python
    install_dependencies
    setup_aliases
    verify_setup
    print_next_steps
}

# Run main if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
