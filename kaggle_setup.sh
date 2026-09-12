#!/usr/bin/env bash
# kaggle_setup.sh — One-shot dependency installer for Kaggle / Colab.
# Run: bash kaggle_setup.sh

set -e

echo "═══════════════════════════════════════════════════════"
echo "  Satellite SR+Seg — Kaggle/Colab setup script"
echo "═══════════════════════════════════════════════════════"

# Upgrade pip first
pip install --upgrade pip --quiet

# Core deep learning
pip install "torch>=2.0" torchvision --quiet

# Image / geospatial
pip install rasterio opencv-python pillow --quiet

# ML utilities
pip install numpy scipy einops pytorch-msssim --quiet

# Segmentation / KAN
pip install efficient-kan --quiet

# App / visualisation
pip install streamlit matplotlib --quiet

echo ""
echo "✅  All dependencies installed."
echo ""
echo "  Quick sanity check:"
echo "    python quick_test.py"
echo ""
echo "  Start training:"
echo "    python train.py"
echo ""
echo "  Run demo app:"
echo "    streamlit run app.py"
