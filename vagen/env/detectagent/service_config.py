from dataclasses import dataclass
from vagen.env.base.base_service_config import BaseServiceConfig

@dataclass
class DetectAgentServiceConfig(BaseServiceConfig):
    name: str = "DetectAgent"
    use_state_reward: bool = False
