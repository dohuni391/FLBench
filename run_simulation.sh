#!/bin/bash

# Script to run federated learning simulations to fill Table 2.
# This script creates a separate, detailed log file for each simulation run,
# sets the correct maximum number of rounds, and terminates the simulation
# once the target accuracy is reached.

# --- Configuration ---
LOG_DIR="logs"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Terminate the python script's process group to ensure all child processes are killed
echo "Killing process group for PID $pid..."
kill -TERM -$pid 2>/dev/null
wait $pid 2>/dev/null

# Force-stop any lingering Ray processes to ensure a clean state for the next run
echo "Cleaning up Ray instance..."
ray stop --force

echo "Starting Federated Learning simulations..."
echo "Individual logs will be saved in the '$LOG_DIR' directory."

# --- Function to run a single simulation ---
run_single_simulation() {
  local dataset="$1"
  local clients="$2"
  local max_rounds="$3" # Added max_rounds as an argument
  local local_epochs=5
  local desired_accuracy=0.90
  local timestamp
  timestamp=$(date -u +"%Y-%m-%d_%H-%M-%S")

  # --- Create a unique log file name for this specific run ---
  local log_file="${LOG_DIR}/log_${dataset}_${clients}clients_${timestamp}.log"

  # This block groups all the commands for a single run.
  # All stdout and stderr from this block will be piped to `tee`,
  # which prints to the console and appends to the specific log file.
  {
    # The command to be executed.
    local cmd="python -u federated_learning.py --dataset $dataset --num_clients $clients --local_epochs $local_epochs --desired_accuracy $desired_accuracy --max_rounds $max_rounds"

    # Execute the command within a coprocess to capture its output
    coproc { $cmd 2>&1; }
    local pid=$!

    # Read from the coprocess's output line by line
    while IFS= read -r line; do
      echo "$line"
      if [[ "$line" == *"--- COMPLETE ---"* ]]; then
        echo "SUCCESS condition met. Terminating simulation for $dataset with $clients clients."
        break
      fi
    done <&"${COPROC[0]}"
  } | tee -a "$log_file" # Pipe all output from the block to the log file and console

  # Terminate the python script's process group to ensure all child processes are killed
  echo "Killing process group for PID $pid..."
  kill -TERM -$pid 2>/dev/null
  wait $pid 2>/dev/null

  # Force-stop any lingering Ray processes to ensure a clean state for the next run
  echo "Cleaning up Ray instance..."
  ray stop --force

  sleep 5 # Brief pause to ensure all resources are fully released before the next run
}

# --- MNIST & CIFAR-10 Runs ---
# Max rounds: 500
# max_rounds_mnist_cifar=500
# for dataset in "mnist" "cifar10"
# do
#   for clients in 10 50 100
#   do
#     run_single_simulation "$dataset" "$clients" "$max_rounds_mnist_cifar"
#   done
# done

# --- FEMNIST Runs ---
# Max rounds: 1200
max_rounds_femnist=1200
for clients in 500 1000
do
  run_single_simulation "femnist" "$clients" "$max_rounds_femnist"
done

echo "All simulations completed."
echo "Check the '$LOG_DIR' directory for individual log files."