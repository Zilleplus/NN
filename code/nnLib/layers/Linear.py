from typing import Any
from nnLib import Module

class Linear(Module):
    in_features: int
    out_features: int

    weights: Any
    bias: Any

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

    
    def forward(self, x):
        ...