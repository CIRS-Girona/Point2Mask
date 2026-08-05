import numpy as np


class Pose:
    def __init__(self) -> None:
        self.T: np.ndarray = None  # Camera to world pose transformation
        self.label: str = None     # File name without extension
