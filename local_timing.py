import argparse
import time
import numpy as np
from utils import get_dataloaders, get_model, train

def main():
    parser = argparse.ArgumentParser(description="Local Training Time Measurement")
    parser.add_argument("--dataset", type=str, required=True, choices=["MNIST", "CIFAR10", "FEMNIST"])
    parser.add_argument("--epochs", type=int, default=5, help="Number of local epochs.")
    parser.add_argument("--clients", type=int, required=True, help="Total number of clients to simulate partitioning.")
    args = parser.parse_args()

    print(f"Measuring local training time for {args.dataset} with {args.clients} clients...")

    # Load data for one client (client 0)
    # Correctly unpack all five return values
    trainloader, _, num_samples, img_key, label_key = get_dataloaders(args.dataset, 0, args.clients)

    # Time 3 runs
    timings = []
    for i in range(3):
        print(f"  Run {i+1}/3...")
        # Re-initialize model for a fair timing run
        model = get_model(args.dataset)
        
        start_time = time.perf_counter()
        # Pass both img_key and label_key to the train function
        train(model, trainloader, epochs=args.epochs, img_key=img_key, label_key=label_key)
        end_time = time.perf_counter()
        timings.append(end_time - start_time)

    mean_time = np.mean(timings)
    print("\n--- Results ---")
    print(f"Dataset: {args.dataset}")
    print(f"Samples / client: {num_samples}")
    print(f"Mean time for {args.epochs} epochs (3 runs): {mean_time:.4f} seconds")

if __name__ == "__main__":
    main()