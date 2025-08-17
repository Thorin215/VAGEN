from vagen.env.base.base_env import BaseEnv
# from vagen.env.svg.svg_utils import (process_and_rasterize_svg, is_valid_svg, load_svg_dataset)
# from vagen.env.svg.score import calculate_total_score
from vagen.env.utils.context_utils import parse_llm_raw_response, convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import DetectAgentEnvConfig
from .tool import ToolCaller

import os
import re
import json
import logging
import random
from PIL import Image
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datasets import Dataset

class DetectAgentEnv(BaseEnv):
    """
    DetectAgentEnv 是一个用于图像伪造检测的环境，继承自 BaseEnv。
    它提供了与图像伪造检测任务相关的功能，包括初始化、重置、执行步骤等。
    """

    def __init__(self, config: DetectAgentEnvConfig):
        super().__init__(config)
        self.config = config

        self.total_reward = 0
        self.reward = 0

        self.prompt_format = self.config.get("prompt_format", "first_prompt")
        self.max_steps = self.config.get("max_steps", 10)
        self.current_step = 0

        self.format_prompt_func = format_prompt[self.prompt_format]

        self.parse_func = PARSE_FUNC_MAP[self.prompt_format]
        self.tool_caller = ToolCaller()
        self.image_path = self.config.get("image_path", "")

        self.conversations = []
        self.conversations.append(
            {
                "role": "user",
                "content": [
                    # {"type":"image", "format": self.prompt_format}
                    # {""}
                ]
            }
        )

    def reset(self, seed=None) -> Dict[Dict, Dict]:
        


    def step(self, action_str: str, dino_model=None, dreamsim_model=None) -> Tuple[Dict, float, bool, Dict]:
        """Execute one step within the environment."""
        self.current_step += 1

        # 解析动作字符串，提取动作类型和参数
        rst = self.parse_func(action_str)

        if rst["answer_content"] and (rst["answer_content"].strip() == "no" or rst["answer_content"].strip() == "yes"):
            # 得到鉴定结果，终止这轮判断
            done = True

        if not rst["tool_content"]:
            # 此轮没有使用工具或未正确输出名字
            return 
        
        tool_name = rst["tool_content"]

        if tool_name:
            tool_result = self.tool_caller.call_tool(
                tool_name,
                {"image_path": self.image_path}
            )

        result_path = self.extract_result_path(tool_result)

        
        # 更新对话记录

        if self.current_step >= self.max_steps:
            self.prompt_format = self.config.get("prompt_format", "final_prompt")
            self.conversations.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "format": self.prompt_format}
                    ]
                }
            )
        else :
            self.prompt_format = self.config.get("prompt_format", "continue_prompt")
            self.conversations.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "format": self.prompt_format}
                    ]
                }
            )

    def extract_result_path(tool_result: Dict) -> str:
        """从工具返回结果中提取图片路径"""
        if not tool_result.get('data'):
            return ""
        
        data = tool_result['data']
        return (
            data.get("heatmap_path") or
            data.get("sharpened_path") or 
            data.get("ela_path") or
            data.get("histogram_path") or
            ""
        )

    def format_prompt(self, **kwargs) -> str:
        """
        Format the prompt based on the current configuration and parameters.
        
        Args:
            **kwargs: Additional parameters for formatting the prompt
            
        Returns:
            str: The formatted prompt string
        """
        return self.format_prompt_func(**kwargs)

    def get_image(self, image_path: str) -> Image.Image:
        """
        Load an image from the specified path.
        
        Args:
            image_path (str): The path to the image file
            
        Returns:
            Image.Image: The loaded image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        return convert_numpy_to_PIL(image_path)
    
    def close(self):
        """
        Close the environment and release any resources.
        """
        # Implement any necessary cleanup here
        pass

    def _render(self, init_obs=False):
        """Render the current state of the environment.

        return all the images and conversations in this step
        """

        if init_obs:
            img = self.image_path
        
        format_prompt_text = 

        
