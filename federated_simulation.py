import argparse
import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Tuple
from flwr.common import NDArrays, Scalar

import torch
from utils import get_model, train, test, get_dataloaders, DEVICE

# Define a Flower client which inherits from NumPyClient.
# NumPyClient is a simple client that handles model parameters as NumPy arrays.
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, dataset_name: str, num_clients: int):
        """Initialize the client with its own unique data partition."""
        self.cid = cid
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.net = get_model(dataset_name)
        
        # Each client loads its own unique partition of the dataset.
        # The client ID (cid) is used to select the correct data shard.
        self.trainloader, self.testloader, self.num_examples, self.img_key, self.label_key = get_dataloaders(
            dataset_name, int(cid), num_clients
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays):
        """Update the local model with parameters received from the server."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        This is the main training method called by the server.
        1. Set model parameters.
        2. Train the model on its local data.
        3. Return the updated model parameters to the server.
        """
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", 5)) # Get number of epochs from server config.
        train(self.net, self.trainloader, epochs=epochs, img_key=self.img_key, label_key=self.label_key)
        return self.get_parameters(config={}), self.num_examples, {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        This method is called by the server for evaluation.
        (Note: In this setup, we use centralized evaluation, so this is not called by the FedAvg strategy).
        """
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader, img_key=self.img_key, label_key=self.label_key)
        return float(loss), self.num_examples, {"accuracy": float(accuracy)}


def client_fn(cid: str, dataset_name: str, num_clients: int) -> FlowerClient:
    """A factory function to create a new FlowerClient instance for a given client ID."""
    return FlowerClient(cid, dataset_name, num_clients)

def main():
    parser = argparse.ArgumentParser(description="Federated Simulation with Flower")
    parser.add_argument("--dataset", type=str, required=True, choices=["MNIST", "CIFAR10", "FEMNIST"])
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5, help="Number of local epochs.")
    parser.add_argument("--target_acc", type=float, default=0.9, help="Target test accuracy to stop.")
    args = parser.parse_args()

    # --- Simulation setup based on the testing plan ---
    if args.clients <= 100:
        datasets_to_run = ["MNIST", "CIFAR10"]
        num_rounds = 500
    else:
        datasets_to_run = ["FEMNIST"]
        num_rounds = 1200
    
    # Allow overriding the plan for specific one-off runs.
    if args.dataset not in datasets_to_run:
        print(f"Warning: The specified dataset {args.dataset} does not match the testing plan for {args.clients} clients.")
        print(f"Proceeding with {args.dataset} as requested.")
        datasets_to_run = [args.dataset]


    for dataset in datasets_to_run:
        print(f"\n--- Running Simulation: {dataset} with {args.clients} clients ---")

        # Function to pass configuration to clients during training.
        def fit_config(server_round: int):
            return {"local_epochs": args.epochs}

        # --- Centralized Evaluation Function ---
        # This function is executed on the SERVER side after each round.
        # It evaluates the globally aggregated model on the full, centralized test set.
        def evaluate_fn(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, Dict[str, Scalar]]:
            model = get_model(dataset)
            # Update the server-side model with the aggregated parameters.
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            
            # Get the full test set for evaluation.
            _, testloader, _, img_key, label_key = get_dataloaders(dataset, 0, args.clients)
            loss, accuracy = test(model, testloader, img_key=img_key, label_key=label_key)
            
            print(f"Round {server_round}: Test Accuracy = {accuracy:.4f}")
            
            # This logic is for early stopping, but the simulation will run for num_rounds regardless.
            if accuracy >= args.target_acc:
                print(f"Target accuracy of {args.target_acc} reached.")
            
            return loss, {"accuracy": accuracy}

        # Configure the FedAvg strategy.
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,           # Use all clients for training in each round.
            fraction_evaluate=0.0,      # Disable client-side evaluation (we use centralized evaluate_fn).
            min_fit_clients=args.clients,
            min_available_clients=args.clients,
            on_fit_config_fn=fit_config,
            evaluate_fn=evaluate_fn,    # Set the centralized evaluation function.
        )

        # Resources to allocate for each client simulation. Helps parallelize on multi-core CPUs or GPUs.
        client_resources = {"num_cpus": 1, "num_gpus": 0.1} if DEVICE.type == "cuda" else {"num_cpus": 1}

        # Start the simulation.
        history = fl.simulation.start_simulation(
            client_fn=lambda cid: client_fn(cid, dataset, args.clients),
            num_clients=args.clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources=client_resources,
        )

        # --- Report Final Results ---
        # Parse the history object to find when the target accuracy was met.
        final_accuracy = 0
        rounds_to_target = -1
        if 'accuracy' in history.metrics_centralized:
            for r, acc in history.metrics_centralized["accuracy"]:
                if acc >= args.target_acc:
                    rounds_to_target = r
                    final_accuracy = acc
                    break # Stop at the first time the target is met.
        
        if rounds_to_target != -1:
            print(f"\nTarget accuracy ({args.target_acc}) reached in {rounds_to_target} rounds.")
            print(f"Final accuracy: {final_accuracy:.4f}")
        else:
            if 'accuracy' in history.metrics_centralized and history.metrics_centralized['accuracy']:
                 final_round, final_acc = history.metrics_centralized["accuracy"][-1]
                 print(f"\nTarget accuracy not reached within {num_rounds} rounds.")
                 print(f"Accuracy at final round {final_round}: {final_acc:.4f}")
            else:
                 print(f"\nTarget accuracy not reached within {num_rounds} rounds, and no accuracy metrics were recorded.")


if __name__ == "__main__":
    main()