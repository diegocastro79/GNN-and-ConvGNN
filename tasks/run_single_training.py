import pickle
from constants.constants import DATA_PATH, MODEL_STATE_PATH, STATISTICS_PATH, PLOTS_PATH, NodeUpdateMethods
import pathlib
from trainer.trainer import Trainer, HyperParameters, test_model
from models.cgnn import ConvGNN
from models.gatnn import GAttNN


def load_data(data_path):
    with open(data_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    method = NodeUpdateMethods.Attention
    hyperparams = HyperParameters(
        learning_rate=1e-2,
        betas=(0.96, 0.98),
        feat_dropout=0.5,
        num_epochs=100,
        dims=[16, 10],
        heads=[8, 4],
        att_dropout=0.1,
    )
    data = load_data(pathlib.Path(DATA_PATH / "graph_data.pkl"))
    trainer = Trainer(
        params=hyperparams,
        data=data,
        gnn=ConvGNN if method == NodeUpdateMethods.Convolution else GAttNN,
        model_estate_path=MODEL_STATE_PATH,
        statistics_path=STATISTICS_PATH,
        plots_path=PLOTS_PATH,
        drop_edges=False,
        method=method,
    )
    trainer.run_epochs(trial=0, save_outcome=False, visualize=True, verbose = True)
    print(f"Test accuracy: {test_model(trainer.model, trainer.data):.4f}")



