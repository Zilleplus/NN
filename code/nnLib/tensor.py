from typing import Tuple
import numpy as np


class Tensor:
    __data: np.ndarray

    def __init__(self, data: np.ndarray):
        self.__data = data

    @staticmethod
    def zeros(shape: Tuple):
        return Tensor(np.zeros(shape))

    def numpy(self):
        return self.__data.copy()

    def __add__(self, other):
        return Tensor(self.__data + other.__data)

    def __sub__(self, other):
        return Tensor(self.__data + other.__data)

    def __mul__(self, other):
        return Tensor(self.__data * other.__data)