import optuna
import pickle
import json
from pathlib import Path
from trainer.trainer import Trainer, HyperParameters, test_model
from models.abstract_model import AbstractGNN
from models.models_utils import save_upload_model_state
from dataclasses import dataclass, asdict
from constants.constants import (
    DATA_PATH, RESULTS_PATH, STATISTICS_PATH, MODEL_STATE_PATH, NUM_LABELS, NUM_FEATURES, NodeUpdateMethods, GnnModelArgs
)
from metrics.metrics import accracy_per_label


@dataclass
class TuningSettings:
    NumTrials: int
    MinEpochs: int
    MaxEpochs: int
    MaxIntLayers: int
    MinIntDim: int
    MaxIntDim: int
    MaxNumHeads: int
    MinNumHeads: int
    Method: NodeUpdateMethods
    DropEdges: bool

    def as_dict(self):
        return {k: v.value if isinstance(v, NodeUpdateMethods) else v for k, v in asdict(self).items()}


class Tuner:
    def __init__(self, settings: TuningSettings, gnn: AbstractGNN):
        self.settings = settings
        self.params_settings_path = self._set_directory_paths(RESULTS_PATH)
        self.model_state_path = self._set_directory_paths(MODEL_STATE_PATH)
        self.statistics_path = self._set_directory_paths(STATISTICS_PATH)
        self._create_directories()
        self.data = self._load_data(Path(DATA_PATH / "graph_data.pkl"))
        self.study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        self.gnn = gnn

    def _set_directory_paths(self, path: Path):
        path = Path(path / self.settings.Method.value)
        return path if not self.settings.DropEdges else Path(path / "drop_edges")

    @staticmethod
    def _load_data(data_path: Path):
        with open(data_path, "rb") as f:
            return pickle.load(f)

    def _create_directories(self):
        self.model_state_path.mkdir(parents=True, exist_ok=True)
        self.statistics_path.mkdir(parents=True, exist_ok=True)
        self.params_settings_path.mkdir(parents=True, exist_ok=True)

    def set_training_parameters(self, trial: optuna.Trial, method: str = "conv"):
        num_layers = trial.suggest_int("num_layers", low=1, high=self.settings.MaxIntLayers)
        dims = [
            trial.suggest_int("dim_"+str(i), low=self.settings.MinIntDim, high=self.settings.MaxIntDim)
            for i in range(num_layers)
        ]
        if self.settings.Method == NodeUpdateMethods.Attention:
            heads = [
                trial.suggest_int("heads_"+str(i), low=self.settings.MinNumHeads, high=self.settings.MaxNumHeads)
                for i in range(num_layers)
            ]
            att_dropout = trial.suggest_float(GnnModelArgs.AttDropout.value, low=0.01, high=0.2)
        else:
            heads, att_dropout = None, None
        return HyperParameters(
            learning_rate=trial.suggest_float("learning_rate", low=1e-5, high=1e-1, log=True),
            betas=(
                trial.suggest_float("beta_1", low=0.95, high=0.9899),
                trial.suggest_float("beta_2", low=0.98, high=0.9999)
            ),
            dims=dims,
            heads=heads,
            num_epochs=trial.suggest_int("num_epochs", low=self.settings.MinEpochs, high=self.settings.MaxEpochs),
            feat_dropout=trial.suggest_float(GnnModelArgs.FeatDropout.value, low=0.3, high=0.7),
            att_dropout=att_dropout,
        )

    def objective(self, trial: optuna.Trial):
        trainer = Trainer(
            params=self.set_training_parameters(trial),
            data=self.data,
            gnn=self.gnn,
            method=self.settings.Method,
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
        with open(Path(self.params_settings_path / "tuning_settings.json"), "w+") as f:
            json.dump(self.settings.as_dict(), f)

    def save_best_model(self):
        best_trial = self.study.best_trial
        params_dict, gnn_args =  self._get_params_dict(best_trial.params)
        # instantiate dummy model with the appropriate architecture
        model = self.gnn(gnn_args)
        # load in the parameters of the best model
        model = save_upload_model_state(
            model=model,
            path=self.model_state_path,
            trial=best_trial.number,
            upload=True
        )
        # save model state as best model
        save_upload_model_state(model=model, path=self.model_state_path, trial=None, upload=False)
        with open(Path(self.params_settings_path / "best_training_params.json"), "w+") as f:
            json.dump(params_dict, f)
        print(f"\nNode classification test accuracy with the best model: {test_model(model, self.data):.4f}\n")
        logits = model(self.data.x, self.data.edge_index)
        acc_per_label = accracy_per_label(logits, self.data.y, self.data.test_mask)
        print(f"Accuracy per label: {acc_per_label}\n")
        print(f"Best training parameters:\n {params_dict}\n")

    def _get_params_dict(self, params):
        gnn_args = dict()
        gnn_args[GnnModelArgs.DimList] = [NUM_FEATURES] + [params[k] for k in sorted(params.keys()) if "dim" in k]
        gnn_args[GnnModelArgs.FeatDropout] = params[GnnModelArgs.FeatDropout.value]
        gnn_args[GnnModelArgs.NumClasses] = NUM_LABELS
        if self.settings.Method == NodeUpdateMethods.Attention:
            gnn_args.update(
                {
                    GnnModelArgs.AttHeadsList: [params[k] for k in sorted(params.keys()) if "heads" in k],
                    GnnModelArgs.AttDropout: params[GnnModelArgs.AttDropout.value],
                }
            )
        else:
            gnn_args.update({GnnModelArgs.DropEdges: self.settings.DropEdges})
        params_dict = {k.value: v for k, v in gnn_args.items()}
        params_dict.update(
            {
                "learning_rate": params["learning_rate"],
                "betas": (params["beta_1"], params["beta_2"]),
            }
        )
        return params_dict, gnn_args





