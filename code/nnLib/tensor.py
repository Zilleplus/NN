from typing import Any, Optional, Tuple
import numpy as np
from nnLib import Function, Add, Mul

class Tensor:
    __data: np.ndarray
    requires: bool
    ctx: Optional[Function]
    grad: Optional['Tensor']


    def __init__(self, data: np.ndarray, requires_grad=True):
        self.__data = data
        self.requires_grad = requires_grad
        self.ctx = None
        self.grad = None

    @property
    def shape(self):
        return self.shape()

    @staticmethod
    def zeros(shape: Tuple):
        return Tensor(np.zeros(shape))

    @staticmethod
    def eye(size: int):
        return Tensor(np.eye(size))

    @staticmethod
    def ones(shape: Tuple):
        return Tensor(np.ones(shape))

    def numpy(self)-> np.ndarray:
        return self.__data.copy()

    def add_data(self, other: 'Tensor'):
        return Tensor(self.__data + other.__data)

    def sub_data(self, other: 'Tensor'):
        return Tensor(self.__data + other.__data)

    def mul_data(self, other: 'Tensor'):
        return Tensor(data=self.__data * other.__data)

    def neg_data(self):
        return Tensor(data=-self.__data)

    def __add__(self, other: 'Tensor'):
        return Add.apply(self, other)

    def __mul__(self, other: 'Tensor'):
        return Mul.apply(self, other)

    def topological_sort(self):
        if self.ctx is None:
            # when you call backward on a variable 
            # that was not yet used in an expression
            # it should no nothing.
            return []
        visited = set() # don't visited the same node twice
        output = []
        def vis(n):
            if n in visited:
                return
            visited.add(n)
            if n.ctx is not None:
                for child in n.ctx.tensors:
                    vis(child)
                output.append(n)
                # if you dont' have ctx, then your an end node.
                # no need to visit you in backprop

        vis(self)

        return output

    def backward(self):
        self.grad = Tensor.backward(self.shape())
        sorted_nodes: list['Tensor'] = self.topological_sort()
        for n in reversed(sorted_nodes):
            # call the Function::backward method
            g = n.ctx.backward(n.grad)
            # set the value onto the children
            for index, (t, grad) in enumerate(zip(n.ctx.tensors, g)):
                if t.requires_grad:
                    t.grad = grad