#!/bin/bash

# ==============================================================================
# Script to measure local computation time for FedAvg experiments.
#
# This script automates the process of running local_timing.py for each
# dataset configuration required to fill Table 1 of the fedavg_timing_plan.md
# document. It captures the stdout from each run, parses the relevant metrics,
# and presents a formatted Markdown summary table.
#
# Usage:
#   ./run_table1_timing.sh
#   (Prints progress to stderr and the final summary to stdout)
#
#   ./run_table1_timing.sh <output_filename.md>
#   (Prints progress to stderr and saves the final summary to the specified file)
#
# Author:      dohuni391
# Date:        2025-07-09 05:09:01 UTC
# ==============================================================================

# 'set -e' ensures that the script will exit immediately if any command fails.
set -e

# Wrap the main logic in a function to allow for easy output redirection.
run_and_summarize() {
    # --- Progress Indicators (sent to stderr) ---
    # By redirecting progress messages to stderr (>&2), they will always
    # appear on the console, even when stdout is being saved to a file.
    echo "--- Starting Local Computation Timing Runs for Table 1 ---" >&2
    echo >&2

    # --- Run and Capture: MNIST ---
    echo "➡️  1/3: Measuring local timing for MNIST (100 clients)..." >&2
    output_mnist=$(python local_timing.py --dataset MNIST --clients 100)
    samples_mnist=$(echo "$output_mnist" | grep "Samples / client" | awk '{print $4}')
    time_mnist=$(echo "$output_mnist" | grep "Mean time" | awk '{print $8}')
    echo "✅ MNIST timing complete." >&2
    echo >&2

    # --- Run and Capture: CIFAR-10 ---
    echo "➡️  2/3: Measuring local timing for CIFAR-10 (100 clients)..." >&2
    output_cifar10=$(python local_timing.py --dataset CIFAR10 --clients 100)
    samples_cifar10=$(echo "$output_cifar10" | grep "Samples / client" | awk '{print $4}')
    time_cifar10=$(echo "$output_cifar10" | grep "Mean time" | awk '{print $8}')
    echo "✅ CIFAR-10 timing complete." >&2
    echo >&2

    # --- Run and Capture: FEMNIST ---
    echo "➡️  3/3: Measuring local timing for FEMNIST (500 clients)..." >&2
    output_femnist=$(python local_timing.py --dataset FEMNIST --clients 500)
    samples_femnist=$(echo "$output_femnist" | grep "Samples / client" | awk '{print $4}')
    time_femnist=$(echo "$output_femnist" | grep "Mean time" | awk '{print $8}')
    echo "✅ FEMNIST timing complete." >&2
    echo >&2

    # --- Final Results Summary (sent to stdout) ---
    # This section's output will be captured by the redirection at the call site.
    echo "--- Results Summary for Table 1 ---"
    echo
    printf "| Dataset               | Samples / client | Time (s) |\n"
    printf "| --------------------- | ---------------- | -------- |\n"
    printf "| MNIST (<= 100 clients) | %-16s | %-8s |\n" "$samples_mnist" "$time_mnist"
    printf "| CIFAR-10 (<= 100)     | %-16s | %-8s |\n" "$samples_cifar10" "$time_cifar10"
    printf "| FEMNIST (> 100)       | %-16s | %-8s |\n" "$samples_femnist" "$time_femnist"
    echo
    echo "--- All local timing runs are complete. ---"
}

# --- Execution Logic ---
# Check if an output file argument ($1) was provided.
if [ -n "$1" ]; then
    # If an argument is provided, call the function and redirect its stdout to the file.
    run_and_summarize > "$1"
    # Also, print a confirmation message to the console (stderr).
    echo -e "\n✅ Results have been saved to '$1'." >&2
else
    # If no argument is provided, just call the function.
    # Its output will go to stdout as usual.
    run_and_summarize
fi