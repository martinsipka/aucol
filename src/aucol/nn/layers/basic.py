import torch

class IndexSelectLayer(torch.nn.Module):
    """
    Layer for index selection used for sequential models.

    Args:
        indices (tensor): tensor of reactants representations
        axis (int): Axis to use indexing along
    Returns:
        (tensor) Tensor after indexing
    """
    def __init__(self, indices):
        super(IndexSelectLayer, self).__init__()

        self.register_buffer('indices', indices, persistent=True)

    def forward(self, inputs, axis=1):
        return torch.index_select(inputs, axis, self.indices)
