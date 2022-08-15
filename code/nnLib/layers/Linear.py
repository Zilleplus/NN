from nnLib import Tensor
from typing import Any
from nnLib import Module

class Linear(Module):
    in_features: int
    out_features: int

    weights: Tensor
    bias: Tensor

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

        # Every output is connected to every input
        # The in_features weights are the rows, and the output the cols
        # This means matmul(weights, input) + bias = output
        self.weights = Tensor.zeros((out_features, in_features), requires_grad=False)
        self.bias = Tensor.constant(value=0, shape=(out_features, 1))
    
    def forward(self, x: Tensor):
        s = x.shape
        assert len(s)>0, "This is a null Tensor, we can't do anything here"

        *_, last  = iter(s)
        assert last == self.in_features, f"We expected {self.in_features} inputs features but got {last}"

        # This operation will require broadcast on the bias.
        # should we do this linearly?
        y = self.weights.matmul(x) + self.bias

        return y