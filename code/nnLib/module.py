from dataclasses import dataclass
from nnLib import Tensor
from typing import TypeVar, overload
from typing import Any

@dataclass
class Parameter:
    name: str
    value: Tensor

class Module:
    training: bool = True
    modules: list['Module'] = []
    parameters: list['Parameter'] = []

    def forward(self, x):
        raise NotImplementedError(f"Forward method of {type(self)}")

    def train(self) -> None:
        """
        set the network in train mode
        """
        self.training = True

    def eval(self) -> None:
        """
        set the network in evaluation mode
        """
        self.training = False

    def register_module(self, module: 'Module'):
        self.modules.append(module)
    
    def register_parameter(self, parameter: Parameter):
        self.parameters.append(parameter)
    
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Module):
            self.register_module(value)
        elif isinstance(value, Tensor):
            self.register_parameter(Parameter(name=name, value=value))
        else:
            # This is some user defined stuff, we just ignore it.
            ...