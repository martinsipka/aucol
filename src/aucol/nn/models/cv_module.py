import numpy as np
import torch
from torch import nn as nn
from torch.autograd import grad

import schnetpack
from schnetpack import nn as L, Properties

class CVModule(nn.Module):
    """
    Predicts Collective variable and its derivative for MD runs.

    Args:
        outnet (callable): Network used for collective variable prediction. Takes schnetpack input
            dictionary as input. Output is now normalized. Cannot be none
        property (str): name of the output property (default: "cv")
        derivative (str or None): Name of property derivative. No derivative
            returned if None. (default: cv_grad)
        create_graph (bool): If False, the graph used to compute the grad will be
            freed. Usually needed only for training and Jacobian calculation (multiD CVs)

    Returns:
        tuple: prediction for CVs and its derivatives if derivatives are not set to None
    """

    def __init__(
        self,
        outnet,
        property="cv",
        derivative="cv_grad",
        create_graph=False,
    ):
        super(CVModule, self).__init__()

        self.create_graph = create_graph
        self.property = property
        self.derivative = derivative
        self.out_net = outnet


    def forward(self, inputs):

        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        y, = self.out_net(inputs)

        # collect results
        result = {self.property: y}

        create_graph = True if self.training or len(y) > 1 else self.create_graph

        if self.derivative is not None:
            cvs_grad = []
            for yi in y:
                dyi, = grad(
                    yi,
                    inputs[Properties.R],
                    create_graph=create_graph,
                    retain_graph=True,
                )
                cvs_grad.append(dyi)
            result[self.derivative] = cvs_grad

        return result
