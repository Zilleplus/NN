from nnLib import Tensor, Module

class Linear(Module):
    in_features: int
    out_features: int

    weights: Tensor
    bias: Tensor

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

    
    def forward(self, x: Tensor):
        ...