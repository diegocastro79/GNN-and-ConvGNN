from tuner.tuner import Tuner, TuningSettings
from models.cgnn import ConvGNN
from models.gatnn import GAttNN
from constants.constants import NodeUpdateMethods

if __name__ == "__main__":
    settings = TuningSettings(
        NumTrials=30,
        MinEpochs=80,
        MaxEpochs=150,
        MaxIntLayers=5,
        MaxIntDim=20,
        MinIntDim=2,
        MaxNumHeads=10,
        MinNumHeads=4,
        DropEdges=True,
        Method=NodeUpdateMethods.Attention,
    )

    tuner = Tuner(settings, gnn=ConvGNN if settings.Method == NodeUpdateMethods.Convolution else GAttNN)
    tuner.optimize()