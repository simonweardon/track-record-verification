#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the track-record verification project.
# Safe to run repeatedly: apt install, venv creation, pip install and the
# reference-cache unpack all converge instead of appending state.
set -euo pipefail

cd "$(dirname "$0")/.."

# System packages: a venv builder for the checked-out Python, plus R and the
# two CRAN libraries the independent-verification twins (r/verify.R, r/construct.R,
# r/decay.R) need. Without R those four tests skip; with it the whole suite runs.
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  python3-venv \
  r-base-core \
  r-cran-data.table \
  r-cran-rglpk

# Python environment.
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# Rebuild the raw data/reference/ layout from the committed ~15 MB compact
# bundle so every loader (and the two price-fingerprint tests) works offline.
# Skips its work when the caches are already present.
.venv/bin/python -m trackrecord compact-unpack

echo "install.sh complete"
