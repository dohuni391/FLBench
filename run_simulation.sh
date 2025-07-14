#!/bin/bash

# ==============================================================================
# Script to run all federated simulations defined in pyproject.toml.
#
# It extracts all federation configurations (e.g., "mnist_10") from the
# pyproject.toml file and then runs `flwr run` for each one.
#
# Usage:
#   ./run_all_simulations.sh
# ==============================================================================

set -e
set -o pipefail

# Create a directory to store the results if it doesn't exist.
mkdir -p results

# --- Extract Experiment Configurations from pyproject.toml ---
# This command uses `grep` to find lines that define a federation,
# then `sed` to clean them up, leaving only the name (e.g., mnist_10).
EXPERIMENTS=$(grep '\[tool.flwr.federations\.' pyproject.toml | sed 's/\[tool.flwr.federations\.//g' | sed 's/\]//g')

# --- Run All Experiments ---
TOTAL_EXPERIMENTS=$(echo "$EXPERIMENTS" | wc -l)
CURRENT_EXPERIMENT=0

echo "--- Starting All Federated Simulations for Table 2 ---"
echo

for exp_name in $EXPERIMENTS; do
    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))

    echo "➡️  (${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}) Running Federation: ${exp_name}..."

    # Run the Flower simulation, passing the app path (.) and the federation name
    # as positional arguments. This is the corrected command format.
    flwr run . "$exp_name"

    echo "✅ (${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}) Complete: ${exp_name}."
    echo
done

echo "--- All simulations are complete. You can now run plot_results.py ---"