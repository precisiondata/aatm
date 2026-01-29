class PipelineBaseClass:
    def __or__(self, other):
        return other(self)

    def __ror__(self, other):
        return self(other)
