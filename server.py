import flwr as fl
import sys
import numpy as np


class CustomAggregationStrategy(fl.server.strategy.FedAvg):

    def __init__(self, num_clusters: int = 2, top_n: int = 3):
        super().__init__(fraction_fit=1.0, fraction_evaluate=0.5, min_fit_clients=3,
                         min_evaluate_clients=3, min_available_clients=3)
        self.num_clusters = num_clusters
        self.top_n = top_n

    def cluster_clients(self, results):
        # Divide clients into `self.num_clusters` clusters based on their hamming percentage

        cluster_size = 100 // self.num_clusters
        clusters = [[] for _ in range(self.num_clusters + 1)]
        hamp = []
        for i in range(len(results)):
            hamp.append(results[i][1].metrics['ham'])
        minimum = min(hamp)
        maximum = max(hamp)
        diff = maximum - minimum
        for i in range(len(results)):
            cur = results[i][1].metrics['ham'] - minimum
            cur = (cur / diff) * 100
            clusters[int(cur // cluster_size)].append(results[i])
        return clusters

    def select_top_models(self, cluster):

        # Select the top `self.top_n` models from each cluster
        cluster.sort(key=lambda x: x[1].metrics['accuracy'])
        return cluster[:min(self.top_n, len(cluster))]

    def aggregate_fit(self,
                      rnd: int,
                      results,
                      failures,
                      ):

        # Cluster clients based on their hamming percentage
        clusters = self.cluster_clients(results)
        print(f"Round {rnd} - Clusters: {[len(c) for c in clusters]}")

        aggregated_weights = []
        for cluster in clusters:

            # Select the top models from the current cluster
            top_models = self.select_top_models(cluster)
            print(f"Round {rnd} - Cluster size: {len(cluster)} - Top {self.top_n} models: {[m[0] for m in top_models]}")

            # Average the selected models' weights
            for i in top_models:
                aggregated_weights.append(i)

        aggregated_weights = super().aggregate_fit(rnd, aggregated_weights, failures)

        # Save the aggregated weights
        if aggregated_weights:
            print(f"Round {rnd} - Saving aggregated weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)

        return aggregated_weights


# Create strategy and run server
strategy = CustomAggregationStrategy(int(sys.argv[2]), int(sys.argv[3]))

# Start Flower server for three rounds of federated learning
fl.server.start_server(
    server_address='localhost:' + str(sys.argv[1]),
    config=fl.server.ServerConfig(num_rounds=1),
    grpc_max_message_length=1024 * 1024 * 1024,
    strategy=strategy
)
