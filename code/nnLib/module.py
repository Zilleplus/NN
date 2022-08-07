class Module:
    training: bool = True

    def forward(self, x):
        raise NotImplementedError()