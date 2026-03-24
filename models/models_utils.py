import torch
from pathlib import Path


def save_upload_model_state(
        model: ConvGNN,
        path: Path,
        trial: int = None,
        upload: bool = False,
        drop_edges: bool = False,
) -> ConvGNN | None:
    if trial is not None:
        file_path = path / f"best_model_trial_{trial}.pt"
    else:
        file_path = path / f"best_model.pt"
    if upload:
        if Path(file_path).exists():
            model.load_state_dict(torch.load(file_path, weights_only=True))
            return model
        raise ValueError(f"The file {path} does not exists!\n")
    torch.save(model.state_dict(), file_path)