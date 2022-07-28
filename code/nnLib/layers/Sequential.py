from typing import List
from nnLib import Module, Tensor

class Sequential(Module):
    __modules: List[Module]

    def __init__(self,modules: List[Module]):
        self.__modules = modules

    def forward(self, x: Tensor):
        for m in self.__modules:
            x = m.forward(x)

        return x