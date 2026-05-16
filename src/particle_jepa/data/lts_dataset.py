from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset


class LearningToSimulateDataset(Dataset):
    """Extension point for DeepMind Learning-to-Simulate particle datasets."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        msg = (
            "LearningToSimulateDataset is a scaffold. Convert TFRecords or exported "
            "arrays into graph-ready tensors, then implement indexing here."
        )
        raise NotImplementedError(msg)
