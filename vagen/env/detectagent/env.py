from vagen.env.base.base_env import BaseEnv
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import DetectAgentEnvConfig
from .tool import ToolCaller
from .prompt import format_prompt as FORMAT_PROMPT_MAP, system_prompt as build_system_prompt

import os
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List

class DetectAgentEnv(BaseEnv):
    """
    DetectAgentEnv 是一个用于图像伪造检测的环境，继承自 BaseEnv。
    它提供了与图像伪造检测任务相关的功能，包括初始化、重置、执行步骤等。
    """

    def __init__(self, config: DetectAgentEnvConfig):
        # BaseEnv has no __init__ signature that accepts config; align with other envs like FrozenLake
        BaseEnv.__init__(self)
        self.config = config

        self.total_reward = 0
        self.reward = 0

        # prompt format lifecycle: first_prompt -> continue_prompt (loop) -> final_prompt
        self.prompt_format: str = "first_prompt"

        self.max_steps = self.config.get("max_steps", 10)
        self.current_step = 0

        # formatter function for current prompt
        self.format_prompt_func = FORMAT_PROMPT_MAP.get(self.prompt_format, FORMAT_PROMPT_MAP.get("first_prompt"))

        # Be robust to map differences across versions by providing a fallback
        self.parse_func = PARSE_FUNC_MAP.get(
            self.prompt_format,
            PARSE_FUNC_MAP.get("no_think", PARSE_FUNC_MAP.get("free_think"))
        )
        self.tool_caller = ToolCaller()
        self.image_path: str = self.config.get("image_path", "")

        # conversation messages: each is {role: 'system'|'user'|'assistant', content: List[{'type':'text','text':...}|{'type':'image','path':...}]}
        self.conversations: List[Dict[str, Any]] = []

        # initialize on creation for convenience
        self._init_conversations()

    # ----------- public API required by BaseEnv -----------
    def system_prompt(self) -> str:
        """Return system prompt for the environment."""
        return build_system_prompt(format=self.prompt_format)

    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """Reset environment and return initial observation and info."""
        self.current_step = 0
        self.total_reward = 0
        self.reward = 0
        self.prompt_format = "first_prompt"
        self.format_prompt_func = FORMAT_PROMPT_MAP[self.prompt_format]
        self.parse_func = PARSE_FUNC_MAP[self.prompt_format]

        self._init_conversations()

        obs = self._build_observation_from_last_user()
        info = {
            "metrics": {},
            "llm_raw_response": "",
            "llm_response": "",
            "conversations": self.conversations,
        }
        return obs, info


    def step(self, action_str: str, dino_model=None, dreamsim_model=None) -> Tuple[Dict, float, bool, Dict]:
        """Execute one step within the environment."""
        self.current_step += 1
        gt_bbox = self.get_gt_bbox()
        print("fuck your mather bitch!")
        # 解析 LLM 输出
        rst = self.parse_func(action_str)

        # 默认未结束
        done = False
        if rst.get("answer_content") and rst["answer_content"].strip().lower() in {"no", "yes"}:
            done = True

        if self.prompt_format == "final_prompt":
            done = True

        # 记录 assistant 的原始响应
        self._append_assistant_message(text=rst.get("llm_raw_response", action_str))
        
        tool_result = None
        result_path = None
        tool_name = (rst.get("tool_content") or "").strip()

        if tool_name and not done:
            # 调用工具
            tool_result = self.tool_caller.call_tool(tool_name, {"image_path": self.image_path})
            result_path = self.extract_result_path(tool_result)

        # 根据状态决定下一个用户提示类型
        if done or self.current_step >= self.max_steps:
            self.prompt_format = "final_prompt"
        else:
            self.prompt_format = "continue_prompt"
        self.format_prompt_func = FORMAT_PROMPT_MAP[self.prompt_format]
        self.parse_func = PARSE_FUNC_MAP[self.prompt_format]

        # 追加下一条用户消息（可能包含工具结果图 + 新提示）
        next_prompt_text = self.format_prompt()
        user_contents: List[Dict[str, Any]] = []

        # 工具结果优先展示，如果没有则仍展示原图
        img_to_show = result_path if result_path else self.image_path
        if img_to_show:
            user_contents.append({"type": "image", "path": img_to_show})
        
        # 如果有工具消息，把工具反馈文字也加进来（便于模型参考）
        if tool_result and tool_result.get("message"):
            user_contents.append({"type": "text", "text": f"[tool:{tool_name}] {tool_result['message']}, {next_prompt_text}"})
        else:
            user_contents.append({"type": "text", "text": next_prompt_text})
        self._append_user_message(contents=user_contents)

        if rst["answer_content"] == "yes":
            self.total_reward += 5.0
            print("Agent confirmed the image is manipulated.")
        elif rst["answer_content"] == "no":
            print("Agent confirmed the image is not manipulated.")

        pred_bbox = list(map(int, rst["region_content"].strip().split(",")))

        step_reward += self.iou(pred_bbox, gt_bbox) if self.iou(pred_bbox, gt_bbox) < 0.3 else 1

        # 简单奖励：格式正确给微小奖励
        step_reward = 1.0 if rst.get("format_correct") else 0.0
        self.total_reward += step_reward

        # 生成新的观测
        obs = self._build_observation_from_last_user()
        info = {
            "metrics": {
                "format_correct": bool(rst.get("format_correct")),
            },
            "llm_raw_response": rst.get("llm_raw_response", action_str),
            "llm_response": rst.get("llm_response", action_str),
            "parsed": rst,
            "conversations": self.conversations,
            "tool_result": tool_result,
        }
        return obs, step_reward, done, info

    def extract_result_path(self, tool_result: Optional[Dict]) -> str:
        """从工具返回结果中提取图片路径"""
        if not tool_result or not tool_result.get('data'):
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
        self.format_prompt_func = FORMAT_PROMPT_MAP[self.prompt_format]
        return self.format_prompt_func()

    def get_gt_bbox(self) -> Optional[List[int]]:
        """Return ground-truth bbox from config if provided, else None.
        The bbox is expected to be [x1,y1,x2,y2] with integer values.
        """
        bbox = self.config.get("gt_bbox") if hasattr(self, "config") else None
        if not bbox:
            return None
        try:
            return [int(b) for b in bbox]
        except Exception:
            return None

    def get_image(self, image_path: str) -> Image.Image:
        """
        Load an image from the specified path.
        
        Args:
            image_path (str): The path to the image file
            
        Returns:
            Image.Image: The loaded image
        """
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return Image.open(image_path).convert("RGB")
    
    def close(self):
        """
        Close the environment and release any resources.
        """
        # Implement any necessary cleanup here
        pass

    def _render(self, init_obs: bool = False) -> Dict[str, Any]:
        """Return all images referenced in the conversation and the full conversation log.

        Returns:
            {
                "image_paths": [str, ...],  # distinct, in chronological order
                "conversations": [...],     # full self.conversations structure
            }
        """
        # [
        #     {"type": "image", "path": c["path"]}
        #     
        # ]
        obs_str= ""

        image_paths: List[str] = []
        for msg in self.conversations:
            if msg.get("role") == "user":
                c = msg.get("content", [])
                if c.get("type") == "image" and c.get("path"):
                    image_paths.append(c["path"])
                content = c.get("text", "")
                obs_str += "User:" + "<image>" + content
            else:
                c = msg.get("content", [])
                # if c.get("type") == "image" and c.get("path"):
                    # image_paths.append(c["path"])
                content = c.get("text", "")
                obs_str += "Assistant:" + content


        # # de-duplicate while preserving order
        # seen = set()
        # deduped_paths: List[str] = []
        # for p in image_paths:
        #     if p not in seen:
        #         seen.add(p)
        #         deduped_paths.append(p)

        # return {
        #     "image_paths": deduped_paths,
        #     "conversations": self.conversations,
        # }
        imgs = [Image.open(p).convert("RGB") for p in image_paths if os.path.exists(p)]
        return {
            "obs_str": obs_str,
            "multi_model_data": {
                "<image>": imgs
            }
        }

    # ----------- helpers -----------
    def _init_conversations(self):
        """Initialize conversations with system and first user message."""
        self.conversations = []
        # system
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": self.system_prompt()}]
        })
        # first user message: original image + first prompt
        contents = []
        if self.image_path:
            contents.append({"type": "image", "path": self.image_path})
        contents.append({"type": "text", "text": self.format_prompt()})
        self._append_user_message(contents=contents)

    def _append_user_message(self, contents: List[Dict[str, Any]]):
        self.conversations.append({"role": "user", "content": contents})

    def _append_assistant_message(self, text: str):
        self.conversations.append({
            "role": "assistant",
            "content": [{"type": "text", "text": text}]
        })

    def _build_observation_from_last_user(self) -> Dict[str, Any]:
        """Compose obs_str and multi_modal_data from the latest user message."""
        if not self.conversations:
            return {"obs_str": "", "multi_modal_data": {"<image>": []}}
        last_user = None
        for msg in reversed(self.conversations):
            if msg.get("role") == "user":
                last_user = msg
                break
        images: List[Image.Image] = []
        text_parts: List[str] = []
        img_placeholders = []
        if last_user:
            for c in last_user.get("content", []):
                if c.get("type") == "image" and c.get("path"):
                    try:
                        images.append(self.get_image(c["path"]))
                        img_placeholders.append("<image>")
                    except Exception:
                        pass
                elif c.get("type") == "text" and c.get("text"):
                    text_parts.append(c["text"])

        obs_str = (" ".join(img_placeholders) + "\n" + "\n".join(text_parts)).strip()
        return {
            "obs_str": obs_str,
            "multi_modal_data": {"<image>": images},
        }

    def iou(bbox1, bbox2) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
