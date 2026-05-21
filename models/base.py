import abc
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    A sequential failure detection model based on a sequence of features.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @abc.abstractmethod
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        args:
        - batch: (dict[str, torch.Tensor]), a dict containing at least:
            - features: torch.Tensor of shape (batch_size, seq_len, input_dim)

        return:
        - mintor loss
        - logs
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_loss(
        self, batch: dict[str, torch.Tensor], weights: list[float] = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
        - batch: (dict[str, torch.Tensor]), a dict containing at least:
            - features: torch.Tensor of shape (batch_size, seq_len, input_dim)
            - valid_masks: torch.Tensor of shape (batch_size, seq_len)
            - labels: torch.Tensor of shape (batch_size,)
        - weights: list[float], the weights for each class

        Return:
        - monitor_loss: torch.Tensor, the monitor loss
        - logs: dict[str, float], a dict of logs
        """

        raise NotImplementedError

    def get_device(self):
        return self._device

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        try:
            self._device = str(next(self.parameters()).device)
        except StopIteration:
            pass
        return module
