"""Flower ServerApp."""

from flwr.common import Context, Parameters, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from .utils import *

def server_fn(context: Context):
    """Define the Flower server logic."""
    # Get properties from context
    dataset_name = context.run_config["dataset_name"]
    num_rounds = context.run_config["num_rounds"]
    fraction_fit = context.run_config["fraction_fit"]
    num_clients = context.run_config["num_clients"]
    batch_size = context.run_config["batch_size"]

    # Define a function for centralized evaluation
    def evaluate_fn(server_round: int, parameters: Parameters, config: dict):
        """Evaluate the model on the centralized test set."""
        # This will only be executed on the server
        data_config = {"img_key": "img", "label_key": "label"}
        if dataset_name in ["mnist", "femnist"]:
            data_config = {"img_key": "image", "label_key": "label"}
            if dataset_name == "femnist":
                data_config["label_key"] = "character"
        
        num_classes = 10 if dataset_name != "femnist" else 62

        model = get_model(dataset_name, num_classes)
        testloader = get_centralized_testloader(dataset_name, batch_size)
        set_weights(model, parameters)
        loss, accuracy = test(model, testloader, data_config)
        return {"loss": loss, "accuracy": accuracy}

    # Initialize model parameters
    initial_net = get_model(dataset_name, 10 if dataset_name != "femnist" else 62)
    initial_parameters = ndarrays_to_parameters(get_weights(initial_net))

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,  # Disable client-side evaluation
        min_available_clients=num_clients,  # Wait for all clients
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,  # Pass the centralized evaluation function
    )

    # Server config
    config = ServerConfig(num_rounds=num_rounds)

    return ServerApp(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(
    server_fn=server_fn,
)