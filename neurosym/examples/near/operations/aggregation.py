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
        """
        Compute the morlet filter for a given sequence.

        :param x: ``(L,)`` sequence of values.
        :return: ``(L,)`` sequence of filter values.
        """
        return torch.exp(-0.5 * torch.pow(self.w * x / self.s, 2)) * torch.cos(
            self.w * x
        )

    def forward(self, batch):
        """
        Apply the morlet filter to the batch.

        :param batch: ``(B, L, D)`` or ``(L, D)`` sequence of values.
        """
        seq_dim = 1 if len(batch.shape) == 3 else 0
        seq_len = batch.shape[seq_dim]
        morlet_filter = (
            self.get_filter(torch.linspace(-self.width, self.width, seq_len))
            .unsqueeze(1)
            .repeat(1, batch.shape[-1])
        )
        return torch.sum(torch.mul(batch, morlet_filter), dim=seq_dim)


def running_agg_torch(
    seq, fn, window_start: callable, window_end: callable, full_window: bool = False
):
    """
    Base function to compute various kinds of running aggregates.

    :param seq: ``(N, L, D)``
    :param fn: ``((N, D) -> (N, D))``
    :param window_start: A callable function f(t:int) -> int that returns the index of the window start.
    :param window_end: A callable function f(t:int) -> int that returns the index of the window end.
    :param full_window: If True, the function will only return the full window aggregates.
    """
    # @TODO: The only reason we aren't allowing 2D is because I don't know how to dynmaically switch between arr[s:e] and arr[:, s:e]
    assert (
        len(seq.shape) == 3
    ), f"Expected 3D tensor with shape (N, L, D), got {seq.shape}"
    N, L, D = seq.shape
    device = seq.device

    # -- 1. build start / end index tensors on the CPU, then move once to GPU
    idxs = torch.arange(L)
    s_idx_np = torch.tensor([max(0, window_start(int(t))) for t in idxs])
    e_idx_np = torch.tensor([min(L - 1, window_end(int(t))) for t in idxs])

    s_idx = s_idx_np.to(device)
    e_idx = e_idx_np.to(device)

    # -- 2. cumulative sum: S[t] = sum_{0..t} seq
    #        shape: (N, L, D)
    cumsum = torch.cumsum(seq, dim=1)

    def expand(v):
        return v.view(1, -1, 1).expand(N, -1, D)

    end_sum = cumsum.gather(1, expand(e_idx))
    start_sum = cumsum.gather(1, expand((s_idx - 1).clamp(min=0)))

    # zero the prefix when the window starts at 0  ↓↓↓
    start_sum = torch.where(
        (s_idx == 0).view(1, -1, 1), torch.zeros_like(start_sum), start_sum
    )

    # -- 4. average over each window
    win_len = (e_idx - s_idx + 1).float().view(1, -1, 1)
    means = (end_sum - start_sum) / win_len  # (N, L, D)

    # -- 5. apply user function in one batched call
    out = fn(means.reshape(-1, D)).reshape(N, L, -1)

    return out[:, -1] if not full_window else out
