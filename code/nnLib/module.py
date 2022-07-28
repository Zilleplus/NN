from nnLib import Tensor


class Module:
    training: bool = True

    def forward(self, x: Tensor):
        raise NotImplementedError()