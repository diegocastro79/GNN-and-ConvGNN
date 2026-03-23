from tuner.tuner import Tuner, TuningSettings

if __name__ == "__main__":
    settings = TuningSettings(
        NumTrials=20,
        MinEpochs=80,
        MaxEpochs=150,
        MaxIntLayers=5,
        MaxIntDim=20,
        MinIntDim=2,
        DropEdges=False,
    )

    tuner = Tuner(settings)
    tuner.optimize()