import pickle
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

if __name__ == "__main__":
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

    # Inspect dataset

    print(f"Dataset name: {dataset.name}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of labels: {dataset.num_classes}\n")

    data = dataset[0]
    with open("data/graph_data.pkl", "wb") as f:
        pickle.dump(data, f)

    num_labels_features = {"num_labels": dataset.num_classes, "num_features": dataset.num_features}
    with open("data/num_labels_features.pkl", "wb") as f:
        pickle.dump(num_labels_features, f)

    print(f"Data: {data}\n")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Fraction of training nodes: {int(data.train_mask.sum()) / data.num_nodes:.2f}")
    print(f"Average node connectivity: {data.num_edges / data.num_nodes: .2f}")

    training_features = data.x[data.train_mask]
    labels = data.y[data.train_mask]
    print(f"A few training node features: {training_features[:5, :10]}")
    print(f"A few node labels: {labels[:5]}")

