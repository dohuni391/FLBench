"""
Measures the local computation time for a single client to complete 5 epochs.

This script reuses the data loading and model definition from the `simulation_app`
package to ensure consistency with the full federated simulations.
"""
import argparse
import time
import numpy as np

# Import the necessary functions from our structured application package
from simulation_app.dataset import get_dataloaders
from simulation_app.model import get_model, train

def main():
    """Parses arguments, runs timing loops, and prints results."""
    parser = argparse.ArgumentParser(
        description="Local Training Time Measurement for FedAvg Plan"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["MNIST", "CIFAR10", "FEMNIST"],
        help="The dataset to use for timing.",
    )
    parser.add_argument(
        "--clients",
        type=int,
        required=True,
        help="Total number of clients to simulate for data partitioning.",
    )
    args = parser.parse_args()

    # --- Setup ---
    print(
        f"Measuring local training time for {args.dataset} with {args.clients} clients..."
    )
    local_epochs = 5

    # Load data for a single representative client (client_id=0)
    # We only need the trainloader and its metadata for this task.
    trainloader, _, img_key, label_key = get_dataloaders(
        dataset_name=args.dataset, client_id=0, num_clients=args.clients
    )
    num_samples = len(trainloader.dataset)

    # --- Timing Loop ---
    timings = []
    print(f"Running 3 trials of {local_epochs} local epochs each...")
    for i in range(3):
        # Re-initialize the model for a fair timing run each time
        model = get_model(dataset_name=args.dataset)

        start_time = time.perf_counter()
        train(model, trainloader, epochs=local_epochs, img_key=img_key, label_key=label_key)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        timings.append(elapsed)
        print(f"  Run {i+1}/3 finished in {elapsed:.4f} seconds.")

    # --- Report Results ---
    mean_time = np.mean(timings)
    print("\n--- Results ---")
    print(f"Dataset: {args.dataset}")
    print(f"Samples / client: {num_samples}")
    print(f"Mean time for {local_epochs} epochs (3 runs): {mean_time:.4f} seconds")

if __name__ == "__main__":
    main()