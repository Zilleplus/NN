class Function:
    @staticmethod
    def forward(ctx):
        ...

    @staticmethod
    def backward(self):
        ...

class LinearFunction(Function):
    ...