import numpy as np
import torch
import flwr as fl
from flwr.common import Parameters, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.client import NumPyClient
import utils as model_utils

class PrefixFedAvg(fl.server.strategy.FedAvg):
    """Federated Averaging with server-side prefix-aware aggregation."""
    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        if not results:
            return None, {}

        # Find the max prefix length sent by any client
        L = max(r.metrics["prefix_len"] for _, r in results if "prefix_len" in r.metrics)
        sum_vec = np.zeros(L, dtype=np.float32)
        cnt_vec = np.zeros(L, dtype=np.float32)

        for _, fit_res in results:
            vec = parameters_to_ndarrays(fit_res.parameters)[0]
            m = fit_res.metrics["prefix_len"]
            n = fit_res.num_examples
            sum_vec[:m] += vec * n
            cnt_vec[:m] += n

        # Fetch previous global vector
        if server_round == 1:
            prev = parameters_to_ndarrays(self.initial_parameters)[0]
        else:
            if not hasattr(self, 'aggregated_parameters_'):
                self.aggregated_parameters_ = self.initial_parameters
            prev = parameters_to_ndarrays(self.aggregated_parameters_)[0]

        # Average where cnt>0, else keep old
        new = np.where(cnt_vec > 0, sum_vec / cnt_vec, prev)
        self.aggregated_parameters_ = ndarrays_to_parameters([new])

        return self.aggregated_parameters_, {}

class PrefixClient(NumPyClient):
    """Flower client implementing prefix-based training and evaluation."""
    def __init__(self, dataset_name, partition_id, num_clients, batch_size, k):
        self.partition_id = partition_id
        self.k = k  # upload fraction (0 < k <= 1)
        self.model = model_utils.get_model(dataset_name)
        self.trainloader, self.valloader, _ = model_utils.get_dataloader(
            dataset_name, partition_id, num_clients, batch_size
        )

    def _flat(self):
        return np.concatenate([p.cpu().numpy().ravel() for p in self.model.parameters()])

    def _set_flat(self, vec):
        offset = 0
        for p in self.model.parameters():
            size = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+size].reshape(p.shape)))
            offset += size

    def get_parameters(self, config):
        return [self._flat()]

    def fit(self, parameters, config):
        self._set_flat(parameters[0])
        local_epochs = config.get("local_epochs", 1)
        model_utils.train(self.model, self.trainloader, local_epochs)
        vec = self._flat()
        cut = int(len(vec) * self.k)
        payload = [vec[:cut]]
        return payload, len(self.trainloader.dataset), {"prefix_len": cut}

    def evaluate(self, parameters, config):
        self._set_flat(parameters[0])
        loss, accuracy = model_utils.test(self.model, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}

def get_prefix_client_fn(dataset_name, num_clients, batch_size):
    def client_fn(cid: str) -> NumPyClient:
        # Make client "2" a straggler, others fast
        k = 0.3 if str(cid) == "2" else 1.0
        return PrefixClient(dataset_name, int(cid), num_clients, batch_size, k)
    return client_fn