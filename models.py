import torch
from torch import nn


# ReLU activation function with LC method
class ReLU_LC(torch.nn.ReLU):
    @staticmethod
    def lc_forward(lc, **kwargs):
        return lc


# GroupSort activation function
class GroupSort(nn.Module):
    def __init__(self, group_size=None, descending=False):
        super(GroupSort, self).__init__()
        self.group_size = group_size
        self.descending = descending
        self.sort_shape = None

    def forward(self, input):
        assert input.shape[1] % self.group_size == 0
        # TODO: cover case where feature depth is not divisible by group size
        if self.sort_shape is None:
            self.sort_shape = [-1, int(input.shape[1] / self.group_size), self.group_size] + list(input.shape[2:])
        output = torch.reshape(input, self.sort_shape)
        output, _ = output.sort(dim=2, descending=self.descending)
        output = torch.reshape(output, input.shape)
        return output

    # Computes Lipschitz constant for each group, based on LC's of incoming features
    def lc_forward(self, lc, softmax=False):
        lc = torch.reshape(lc, self.sort_shape)

        # Uses either a real Lipschitz constant with the max function or a softer approximate Lipschitz constant using
        # the softmax function (for potentially smoother gradients)
        if softmax:
            weights = nn.functional.softmax(lc, dim=2)
            groups = weights*lc
            groups = torch.sum(groups, dim=2, keepdim=True)
        else:
            groups = torch.max(lc, dim=2, keepdim=True).values

        # Replicates group LC's for each feature
        lc = groups + 0*lc
        lc = lc.reshape([-1])
        return lc


class ShallowNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, output_size=10, activation=None):
        super(ShallowNet, self).__init__()
        if activation is None:
            self.activation = GroupSort(group_size=10)
        else:
            self.activation = activation()
        self.inlayer = nn.Linear(input_size, hidden_size)
        self.outlayer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.reshape(x, [x.shape[0], -1])
        x = self.inlayer(x)
        x = self.activation(x)
        x = self.outlayer(x)
        return x

    # Computes a differentiable Lipshcitz constant for each output of the net
    def get_lc(self, softmax=False):
        lc = torch.sum(torch.abs(self.inlayer.weight), [i+1 for i in range(self.inlayer.weight.ndim-1)])
        lc = self.activation.lc_forward(lc, softmax=softmax)
        lc = torch.sum(torch.abs(self.outlayer.weight) * lc.reshape(1, -1), [i + 1 for i in range(self.outlayer.weight.ndim - 1)])
        return lc


# Normalizes weight matrices to be 1-Lipshcitz
def lc_init(m):
    if isinstance(m, nn.Linear):
        mags = torch.sum(torch.abs(m.weight), dim=1, keepdims=True)
        with torch.no_grad():
            m.weight = nn.Parameter(m.weight/mags)
