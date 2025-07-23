#!/bin/bash

# This script runs the local computation timing experiments
# to populate the results table. It runs the python script
# for each dataset configuration and pipes the output to a
# file while also showing it on the console.

# Create a file to store the results
RESULTS_FILE="timing_results.txt"
> $RESULTS_FILE # Clear the file if it exists

# Run experiments and save results
echo "Running timing experiment for MNIST..." | tee -a $RESULTS_FILE
python local_timing.py --dataset mnist --clients 100 --epochs 5 --runs 3 | tee -a $RESULTS_FILE

echo -e "\n----------------------------------------\n" | tee -a $RESULTS_FILE

echo "Running timing experiment for CIFAR-10..." | tee -a $RESULTS_FILE
python local_timing.py --dataset cifar10 --clients 100 --epochs 5 --runs 3 | tee -a $RESULTS_FILE

echo -e "\n----------------------------------------\n" | tee -a $RESULTS_FILE

echo "Running timing experiment for FEMNIST..." | tee -a $RESULTS_FILE
# For FEMNIST, we simulate a partition for >100 clients. We use 200.
python local_timing.py --dataset femnist --clients 200 --epochs 5 --runs 3 | tee -a $RESULTS_FILE

echo -e "\nAll experiments complete. Results saved to $RESULTS_FILE"