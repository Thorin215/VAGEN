from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from vagen.env.base.base_service import BaseService
from vagen.server.serial import serialize_observation
from .env import DetectAgentEnv
from .env_config import DetectAgentEnvConfig
from .service_config import DetectAgentServiceConfig


class DetectAgentService(BaseService):
    """Service wrapper for DetectAgentEnv with batch APIs."""

    def __init__(self, config: DetectAgentServiceConfig):
        # No BaseService __init__; keep local state
        self.environments: Dict[str, DetectAgentEnv] = {}
        self.env_configs: Dict[str, DetectAgentEnvConfig] = {}
        self.config = config
        self.max_workers = getattr(config, "max_workers", 10)

    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """Create multiple environments; skip and log failures instead of raising."""
        def create_single_env(env_id: Any, cfg: Dict[str, Any]):
            try:
                env_config_dict = cfg.get('env_config', {})
                print("env_config_dict:", env_config_dict)
                env_config = DetectAgentEnvConfig(**env_config_dict)
                env = DetectAgentEnv(env_config)
                return env_id, (env, env_config), None
            except Exception as e:
                return env_id, None, str(e)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(create_single_env, env_id, cfg): env_id for env_id, cfg in ids2configs.items()}
            for future in as_completed(futures):
                env_id = futures[future]
                try:
                    env_id, result, error = future.result()
                except Exception as e:
                    print(f"Error creating detectagent env {env_id}: {e}")
                    continue
                if error:
                    print(f"Error creating detectagent env {env_id}: {error}")
                    continue
                env, env_config = result
                self.environments[env_id] = env
                self.env_configs[env_id] = env_config

    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        results: Dict[Any, Tuple[Any, Any]] = {}
        for env_id, seed in ids2seeds.items():
            env = self.environments.get(env_id)
            if env is None:
                results[env_id] = ({}, {"error": f"Environment {env_id} not found"})
                continue
            try:
                observation, info = env.reset(seed=seed)
                serialized_observation = serialize_observation(observation)
                results[env_id] = (serialized_observation, info)
            except Exception as e:
                results[env_id] = ({}, {"error": str(e)})
        return results

    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        results: Dict[Any, Tuple[Dict, float, bool, Dict]] = {}
        for env_id, action in ids2actions.items():
            env = self.environments.get(env_id)
            if env is None:
                results[env_id] = ({}, 0.0, True, {"error": f"Environment {env_id} not found"})
                continue
            try:
                observation, reward, done, info = env.step(action)
                serialized_observation = serialize_observation(observation)
                results[env_id] = (serialized_observation, reward, done, info)
            except Exception as e:
                results[env_id] = ({}, 0.0, True, {"error": str(e)})
        return results

    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        results: Dict[Any, float] = {}
        for env_id in env_ids:
            env = self.environments.get(env_id)
            if env is None:
                results[env_id] = 0.0
                continue
            try:
                results[env_id] = env.compute_reward()
            except Exception:
                results[env_id] = 0.0
        return results

    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        results: Dict[Any, str] = {}
        for env_id in env_ids:
            env = self.environments.get(env_id)
            if env is None:
                results[env_id] = ""
                continue
            try:
                results[env_id] = env.system_prompt()
            except Exception:
                results[env_id] = ""
        return results

    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        if env_ids is None:
            env_ids = list(self.environments.keys())
        for env_id in env_ids:
            env = self.environments.get(env_id)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
