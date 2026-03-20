import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from models.cgnn import ConvGNN, save_upload_model_state
from tqdm import tqdm
from constants.constants import DATA_PATH, NUM_FEATURES, NUM_LABELS
from metrics.metrics import entropy_loss, global_accuracy, accracy_per_label
from pathlib import Path
from dataclasses import dataclass
import json
import pickle
import matplotlib.pyplot as plt


def get_num_feature_labels():
    with open(DATA_PATH / "num_feature_labels.pkl", "rb") as f:
        feat_labels_dict = pickle.load(f)
    return feat_labels_dict["num_features"], feat_labels_dict["num_labels"]


def test_model(model: ConvGNN, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        return global_accuracy(logits, data.y, data.test_mask)[0]


@dataclass
class HyperParameters:
    learning_rate: float
    betas: tuple[float, float]
    dims: list[int] # contains the intermediate layer dimensions
    drop: float
    num_epochs: int

class Trainer(nn.Module):
    def __init__(
            self,
            params: HyperParameters,
            data,
            gnn: ConvGNN,
            model_estate_path: Path,
            statistics_path: Path,
            plots_path: Path | None = None,
            drop_edges: bool = False):
        super().__init__()
        self.params = params
        self.data = data
        self._extract_data()
        self.gnn = gnn
        self.model_estate_path = model_estate_path
        self.statistics_path = statistics_path
        self.drop_edges = drop_edges
        self.plots_path = self._set_directory_path(plots_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _set_directory_path(self, path: Path | None):
        if path is None:
            return None
        return path if not self.drop_edges else Path(path / "drop_edges")

    def _extract_data(self):
        self.x = self.data.x
        self.y = self.data.y
        self.edge_index = self.data.edge_index

    def _init_model(self):
        dim_list = [NUM_FEATURES] + self.params.dims
        self.model = self.gnn(
            dim_list=dim_list, num_classes=NUM_LABELS, drop=self.params.drop, drop_edges=self.drop_edges
        ).to(self.device)

    def _init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params.learning_rate,
            betas=self.params.betas
        )

    def gradient_step(self):
        self.optimizer.zero_grad()
        logits = self.model(self.x, self.edge_index)
        loss = entropy_loss(logits, self.y, self.data.train_mask)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run_epochs(self, trial: int, save_outcome: bool = True, visualize: bool = False, verbose: bool = False):
        self._init_model()
        self._init_optimizer()
        self.model.train()
        train_loss = []
        val_loss = []
        acc_values = []
        iterable = tqdm(range(self.params.num_epochs), desc=f"Training trial {trial}, with dropping edges={self.drop_edges}")
        for epoch in iterable:
            train_loss.append(self.gradient_step())
            if epoch % 20 == 0:
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(self.x, self.edge_index)
                    val_loss.append(entropy_loss(logits, self.y, self.data.val_mask).item())
                    acc_values.append(global_accuracy(logits, self.y, self.data.val_mask)[0])
                self.model.train()
                if verbose:
                    print("\n" + 10 * "#" + f"epoch: {epoch}" + 10 * "#")
                    print(f"\nTraining loss for epoch {epoch}: {train_loss[-1]}")
                    print(f"Validation loss for epoch {epoch}: {val_loss[-1]}")
                    print(f"Classification accuracy for epoch {epoch}: {acc_values[-1]}")
        if save_outcome:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(self.x, self.edge_index)
                acc_per_label = accracy_per_label(logits, self.y, self.data.val_mask)
            with open(Path(self.statistics_path / f"statistics_trial_{trial}.json"), "w") as f:
                json.dump(
                    {
                        "train losses": train_loss,
                        "validation losses": val_loss,
                        "accuracies": acc_values,
                        "accuracy per label": acc_per_label,
                    },
                    f
                )
            self.model.train()
            save_upload_model_state(model=self.model, path=self.model_estate_path, trial=trial, upload=False)
        if visualize and self.plots_path is not None:
            self.visualize(train_loss, val_loss, acc_values, trial)
        return acc_values[-1]

    def visualize(self, train_loss, val_loss, acc_values, trial):
        self.plots_path.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(ncols=2)
        ax[0].plot(range(1, len(train_loss) + 1), train_loss, label="train loss")
        ax[0].legend()
        ax[0].set(xlabel="Epochs", ylabel="Loss")
        x = [20*i for i in range(1, len(val_loss) + 1)]
        ax[1].plot(x, val_loss, label="validation loss")
        ax[1].legend()
        ax[1].set(xlabel="Epochs")
        fig.savefig(Path(self.plots_path / f"losses_{trial}.png"))
        plt.close(fig)
        fig, ax = plt.subplots()
        ax.plot(x, acc_values)
        ax.set_title("Validation accuracy")
        ax.set_xlabel("Epochs")
        fig.savefig(Path(self.plots_path / f"val_accuracy_{trial}.png"))
        plt.close(fig)

