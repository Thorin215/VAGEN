from vagen.env.base import BaseService
from .env import DetectAgentEnv
from .env_config import DetectAgentEnvConfig

class DetectAgentService(BaseService):
    def __init__(self, config: DetectAgentEnvConfig):
        super().__init__(config)
        self.env = DetectAgentEnv(config)
