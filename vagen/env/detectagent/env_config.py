from dataclasses import dataclass, field
from vagen.env.base import BaseEnvConfig

@dataclass
class DetectAgentEnvConfig(BaseEnvConfig):
    name: str = "DetectAgent"
    max_steps: int = 100
