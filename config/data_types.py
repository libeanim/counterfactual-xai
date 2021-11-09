from typing import TypedDict, Union

class Resnet50NewConfig(TypedDict):
    id: str
    random_seed: int
    model: str
    k: int
    learning_rate: float
    momentum: float
    epochs: int
    log_steps: int
    weight_decay: float
    batch_size: int
    base_model: Union[str, None]
    scheduler: Union[str, None]
    milestones: Union[list[int], None]
    gamma: float
    freeze_backbone: bool
    init_readout: Union[str, None]