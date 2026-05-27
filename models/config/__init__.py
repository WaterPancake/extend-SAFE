"""Model config dataclasses for SAFE-style benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    input_dim: int = 4096
    hidden_dim: int = 256
    n_layers: int = 1
    dropout: float = 0.0
    n_epochs: int = 1000
    batch_size: int = 512
    optimizer: str = "adam"
    lr: float = 1e-3
    lr_step_size: int = 300
    lr_gamma: float = 1.0
    weight_decay: float = 1e-2
    warmup_steps: int = 0
    lambda_success: float = 1.0
    lambda_fail: float = 1.0
    lambda_reg: float = 1.0
    cumsum: bool = False
    running_mean: bool = False
    loss_type: str = "bce"
    threshold: float = 1.0
    use_time_weighting: bool = False


@dataclass
class SafeMLPConfig(ModelConfig):
    name: str = "safe_mlp"
    n_layers: int = 2
    cumsum: bool = False
    loss_type: str = "bce"
    final_activation: str = "sigmoid"


@dataclass
class SafeLSTMConfig(ModelConfig):
    name: str = "safe_lstm"
    n_layers: int = 1
    cumsum: bool = False
    loss_type: str = "bce"


LstmModelConfig = SafeLSTMConfig
