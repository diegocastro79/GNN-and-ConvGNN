import pickle
from constants.constants import DATA_PATH, MODEL_STATE_PATH, STATISTICS_PATH, PLOTS_PATH
import pathlib
from trainer.trainer import Trainer, HyperParameters
from models.cgnn import ConvGNN


def load_data(data_path):
    with open(data_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    hyperparams = HyperParameters(
        learning_rate=1e-2,
        betas=(0.96, 0.98),
        drop=0.5,
        num_epochs=100,
        dims=[16],
    )
    data = load_data(pathlib.Path(DATA_PATH / "graph_data.pkl"))
    trainer = Trainer(
        params=hyperparams,
        data=data,
        gnn=ConvGNN,
        model_estate_path=MODEL_STATE_PATH,
        statistics_path=STATISTICS_PATH,
        plots_path=PLOTS_PATH,
    )
    trainer.run_epochs(trial=0, save_outcome=False, visualize=True)
    print(f"Test accuracy: {trainer.test_model()}")



