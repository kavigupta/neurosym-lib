"""
This file contains advanced operations used for aggregating a sequence of features. (ie: [B, S, D] -> [B, D])
Most of these operators are detailed in https://arxiv.org/pdf/2106.06114
"""
import torch

class SymmetricMorletFilter(torch.nn.Module):
    """
    A parametric aggregation function based on a symmetric morlet wavelet.
    Refer to https://arxiv.org/pdf/2106.06114 for a detailed discussion.
    """
    def __init__(self, width=torch.pi):  # noqa: E741
        super().__init__()
        self.width = width
        self.w = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.s = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))

    def get_filter(self, x):
        return (torch.exp(-0.5 * torch.pow(self.w * x / self.s, 2)) * torch.cos(self.w * x))

    def forward(self, batch):
        seq_dim = 1 if len(batch.shape) == 3 else 0
        seq_len = batch.shape[seq_dim]
        morlet_filter = self.get_filter(torch.linspace(-self.width, self.width, seq_len)).unsqueeze(1).repeat(1, batch.shape[-1])
        return torch.sum(torch.mul(batch, morlet_filter), dim=seq_dim)


def running_agg_torch(seq, fn, window_start: callable, window_end: callable):
    """
    Base function to compute various kinds of running aggregates.

    :param seq: ``(N, L, D)``
    :param fn: ``((N, D) -> (N, D))``
    :param window_start: A callable function f(t:int) -> int that returns the index of the window start.
    :param window_end: A callable function f(t:int) -> int that returns the index of the window end.
    """
    # @TODO: The only reason we aren't allowing 2D is because I don't know how to dynmaically switch between arr[s:e] and arr[:, s:e]
    assert len(seq.shape) == 3, f"Expected 3D tensor with shape (N, L, D), got {seq.shape}"
    seq_len = seq.shape[1]
    aggs = []
    for t in range(seq_len):
        start = max(0, window_start(t))
        end = min(seq_len, window_end(t))
        window = seq[:, start:end]
        running_agg = torch.mean(window, dim=1)
        aggs.append(fn(running_agg))
    out = torch.stack(aggs, dim=1)
    return out[:, -1]
