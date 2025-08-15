from dataclasses import dataclass
from vagen.env.base import BaseServiceConfig

@dataclass
class DetectAgentServiceConfig(BaseServiceConfig):
    name: str = "DetectAgent"
