import optuna
import pickle
import json
from pathlib import Path
from trainer.trainer import Trainer, HyperParameters, test_model
from models.cgnn import ConvGNN, save_upload_model_state
from dataclasses import dataclass, asdict
from constants.constants import DATA_PATH, STATISTICS_PATH, MODEL_STATE_PATH, NUM_LABELS, NUM_FEATURES
from metrics.metrics import accracy_per_label


@dataclass
class TuningSettings:
    NumTrials: int
    MinEpochs: int
    MaxEpochs: int
    MaxIntLayers: int
    MinIntDim: int
    MaxIntDim: int
    DropEdges: bool

    def as_dict(self):
        return asdict(self)


class Tuner:
    def __init__(self, settings: TuningSettings):
        self.settings = settings
        self.model_state_path = self._set_directory_paths(MODEL_STATE_PATH)
        self.statistics_path = self._set_directory_paths(STATISTICS_PATH)
        self._create_directories()
        self.data = self._load_data(Path(DATA_PATH / "graph_data.pkl"))
        self.study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    def _set_directory_paths(self, path: Path):
        return path if not self.settings.DropEdges else Path(path / "drop_edges")

    @staticmethod
    def _load_data(data_path: Path):
        with open(data_path, "rb") as f:
            return pickle.load(f)

    def _create_directories(self):
        self.model_state_path.mkdir(parents=True, exist_ok=True)
        self.statistics_path.mkdir(parents=True, exist_ok=True)

    def set_training_parameters(self, trial: optuna.Trial):
        num_layers = trial.suggest_int("num_layers", low=1, high=self.settings.MaxIntLayers)
        dims = [
            trial.suggest_int("dim_"+str(i), low=self.settings.MinIntDim, high=self.settings.MaxIntDim)
            for i in range(num_layers)
        ]
        return HyperParameters(
            learning_rate=trial.suggest_float("learning_rate", low=1e-5, high=1e-1, log=True),
            betas=(
                trial.suggest_float("beta_1", low=0.95, high=0.9899),
                trial.suggest_float("beta_2", low=0.98, high=0.9999)
            ),
            dims=dims,
            num_epochs=trial.suggest_int("num_epochs", low=self.settings.MinEpochs, high=self.settings.MaxEpochs),
            drop=trial.suggest_float("drop", low=0.3, high=0.7)
        )

    def objective(self, trial: optuna.Trial):
        trainer = Trainer(
            params=self.set_training_parameters(trial),
            data=self.data,
            gnn=ConvGNN,
            model_estate_path=self.model_state_path,
            statistics_path=self.statistics_path,
            drop_edges=self.settings.DropEdges,
        )
        try:
            metric_val = trainer.run_epochs(trial=trial.number, save_outcome=True, visualize=False, verbose = False)
        except Exception as e:
            metric_val = 1e-10
            print(f"An exception {e} occurred during trial {trial.number}\n")
        return metric_val

    def optimize(self):
        self.study.optimize(self.objective, n_trials=self.settings.NumTrials)
        self.save_best_model()
        with open(Path(f"../results/tuning_settings_drop_edges_{self.settings.DropEdges}.json"), "w+") as f:
            json.dump(self.settings.as_dict(), f)

    def save_best_model(self):
        best_trial = self.study.best_trial
        params_dict =  self._get_params_dict(best_trial.params)
        # instantiate dummy model with the appropriate architecture
        model = ConvGNN(
            dim_list=[NUM_FEATURES] + params_dict["internal_dims"],
            num_classes=NUM_LABELS,
            drop=params_dict["drop"],
        )
        # load in the parameters of the best model
        model = save_upload_model_state(
            model=model,
            path=self.model_state_path,
            trial=best_trial.number,
            upload=True
        )
        # save model state as best model
        save_upload_model_state(model=model, path=self.model_state_path, trial=None, upload=False)
        with open(Path(f"../results/best_training_params_drop_edges_{self.settings.DropEdges}.json"), "w+") as f:
            json.dump(params_dict, f)
        print(f"\nNode classification test accuracy with the best model: {test_model(model, self.data):.4f}\n")
        logits = model(self.data.x, self.data.edge_index)
        acc_per_label = accracy_per_label(logits, self.data.y, self.data.test_mask)
        print(f"Accuracy per label: {acc_per_label}\n")
        print(f"Best training parameters:\n {params_dict}\n")

    @staticmethod
    def _get_params_dict(params):
        params_dict = dict()
        params_dict["internal_dims"] = [params[k] for k in sorted(params.keys()) if "dim" in k]
        params_dict["learning_rate"] = params["learning_rate"]
        params_dict["betas"] = (params["beta_1"], params["beta_2"])
        params_dict["drop"] = params["drop"]
        return params_dict





