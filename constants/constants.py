from pathlib import Path
import pickle

DATA_PATH = Path("../dataset/data")
MODEL_STATE_PATH = Path("../results/model_states")
STATISTICS_PATH = Path("../results/statistics")
PLOTS_PATH = Path("../results/plots")

def get_num_feature_labels():
    with open("../dataset/data/num_labels_features.pkl", "rb") as f:
        feat_labels_dict = pickle.load(f)
    return feat_labels_dict["num_features"], feat_labels_dict["num_labels"]

NUM_FEATURES, NUM_LABELS = get_num_feature_labels()