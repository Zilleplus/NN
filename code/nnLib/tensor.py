from typing import Any, Optional, Tuple, overload
import numpy as np
from nnLib import Function, Add, Mul, MatMul


class Tensor:
    __data: np.ndarray
    requires: bool
    ctx: Optional[Function]
    grad: Optional['Tensor']

    @overload
    def __init__(self, data: np.ndarray, requires_grad=True):
        ...

    @overload
    def __init__(self, data: 'Tensor', requires_grad=True):
        ...

    def __init__(self, data: Any, requires_grad=True):
        if isinstance(data, np.ndarray):
            # If you pass it a numpy array, take ownership
            self.__data = data
        elif isinstance(data, Tensor):
            # if you pass it a Tensor object -> copy ctor
            self.__data = data.__data.copy()

        self.requires_grad = requires_grad
        self.ctx = None  # used in backprop
        if(requires_grad):
            self.grad = Tensor.zeros(data.shape, requires_grad=False)
        else:
            self.grad = None

    @property
    def shape(self):
        return self.__data.shape

    @staticmethod
    def zeros(shape: Tuple, requires_grad=True):
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def eye(size: int, requires_grad=True):
        return Tensor(np.eye(size), requires_grad=requires_grad)

    @staticmethod
    def constant(value: float, shape: Tuple):
        return Tensor(np.ones(shape=shape)*value, requires_grad=False)

    @staticmethod
    def ones(shape: Tuple, requires_grad=True):
        return Tensor(np.ones(shape), requires_grad=requires_grad)

    @staticmethod
    def from_list(values, requires_grad=True):
        return Tensor(np.array(values), requires_grad=requires_grad)

    def like_zeros(self):
        return Tensor.zeros(shape=self.shape, requires_grad=self.requires_grad)

    def __hash__(self) -> int:
        return id(self)

    def numpy(self) -> np.ndarray:
        return self.__data.copy()

    def add_data(self, other: 'Tensor'):
        return Tensor(self.__data + other.__data,
                      requires_grad=self.requires_grad or other.requires_grad)

    def sub_data(self, other: 'Tensor'):
        return Tensor(self.__data + other.__data,
                      requires_grad=self.requires_grad or other.requires_grad)

    def mul_data(self, other: 'Tensor'):
        return Tensor(data=self.__data * other.__data,
                      requires_grad=self.requires_grad or other.requires_grad)

    def neg_data(self):
        return Tensor(data=-self.__data,
                      requires_grad=self.requires_grad)

    def matmul_data(self, other: 'Tensor') -> 'Tensor':
        (*_ , num_rows_l, num_cols_l) = self.shape
        (*_, num_rows_r, num_cols_r) = other.shape
        
        assert num_cols_l == num_rows_r, "Invalid matmul format"
        
        return Tensor(self.__data*other.__data, requires_grad=False)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        return all(self.__data == other.__data)

    def __add__(self, other: 'Tensor'):
        return Add.apply(self, other)

    def __mul__(self, other: 'Tensor'):
        return Mul.apply(self, other)

    def matmul(self, other: 'Tensor'):
        return MatMul.apply(self, other)

    def topological_sort(self):
        if self.ctx is None:
            # when you call backward on a variable
            # that was not yet used in an expression
            # it should no nothing.
            return []
        visited = set()  # don't visited the same node twice
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
        self.grad = Tensor.ones(self.shape)
        sorted_nodes: list['Tensor'] = self.topological_sort()
        for n in reversed(sorted_nodes):
            # call the Function::backward method
            g = n.ctx.backward(n.grad)
            # set the value onto the children
            for index, (t, grad) in enumerate(zip(n.ctx.tensors, g)):
                if t.requires_grad:
                    t.grad = t.grad.add_data(grad)
