"""Flower ServerApp."""

import json
import os
import time

import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.server import Server
from flwr.server.strategy import FedAvg
from flwr.server.callback import Callback
from flwr.server.history import History

from .model import get_model, get_weights
from .strategy import gen_evaluate_fn

class ResultsCallback(Callback):
    """Callback to save results at the end of the simulation."""
    def __init__(self, federation_name: str):
        self.start_time = time.time()
        self.federation_name = federation_name

    def on_driver_finished(self, driver, history: History) -> None:
        """Save results to a JSON file when the simulation ends."""
        history.metrics_centralized["start_time"] = [self.start_time]
        output_filename = f"results/results_{self.federation_name}.json"
        os.makedirs("results", exist_ok=True)
        with open(output_filename, "w") as f:
            json.dump(history.metrics_centralized, f, indent=4)
        print(f"\nResults for {self.federation_name} saved to {output_filename}")

def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # 1. Get configuration from the context
    run_config = context.run_config
    federation_name = context.federation_name
    print("--- Experiment Config ---")
    print(f"  Federation: {federation_name}")
    print(f"  Run Config: {json.dumps(run_config, indent=2)}")
    print("-------------------------")

    # 2. Extract parameters
    dataset_name = str(run_config["dataset"])
    num_clients = int(run_config["num_clients"])
    num_rounds = int(run_config["num_rounds"])
    
    # 3. Initialize model and strategy
    initial_parameters = ndarrays_to_parameters(get_weights(get_model(dataset_name)))
    evaluate_fn = gen_evaluate_fn(dataset_name)
    
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
    )
    
    # 4. Create Server and ServerConfig
    server = Server(strategy=strategy, callbacks=[ResultsCallback(federation_name)])
    config = ServerConfig(num_rounds=num_rounds)
    
    # 5. Return the ServerApp's components
    return server, config

# The ServerApp, which `flwr run` will instantiate using the `server_fn`.
app = ServerApp(server_fn=server_fn)