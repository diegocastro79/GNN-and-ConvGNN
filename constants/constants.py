from pathlib import Path
import pickle
from enum import Enum

DATA_PATH = Path("../dataset/data")
RESULTS_PATH = Path("../results")
MODEL_STATE_PATH = Path("../results/model_states")
STATISTICS_PATH = Path("../results/statistics")
PLOTS_PATH = Path("../results/plots")

def get_num_feature_labels():
    with open("../dataset/data/num_labels_features.pkl", "rb") as f:
        feat_labels_dict = pickle.load(f)
    return feat_labels_dict["num_features"], feat_labels_dict["num_labels"]

NUM_FEATURES, NUM_LABELS = get_num_feature_labels()

class NodeUpdateMethods(Enum):
    Convolution = "conv"
    Attention = "att"

class GnnModelArgs(Enum):
    DimList = "dim_list"
    AttHeadsList = "att_heads_list"
    NumClasses = "num_classes"
    DropEdges = "drop_edges"
    FeatDropout = "feat_dropout"
    AttDropout = "att_dropout"