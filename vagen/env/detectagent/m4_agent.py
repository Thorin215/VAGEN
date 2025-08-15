import json
import base64
from typing import Dict, List, Optional, Tuple
import os
import re
from PIL import Image

from vagen.inference.model_interface.base_model import BaseModelInterface
try:
    from vagen.inference.model_interface.vllm import VLLMModelInterface, VLLMModelConfig
except Exception:
    VLLMModelInterface = None  # type: ignore
    VLLMModelConfig = None  # type: ignore

from .tool import ToolCaller
from .prompt import first_prompt, continue_prompt, final_prompt


class LocalVLMSession:
    """
    本地推理会话：
    - 维护 messages 历史（Qwen 训练同款消息格式）
    - 使用 VAGEN 的 model_interface（推荐 vLLM）进行本地推理
    - 支持图像 + 文本（通过 <image> + multi_modal_data）
    """

    def __init__(self, model_interface: Optional[BaseModelInterface] = None) -> None:
        self.messages: List[Dict] = []
        if model_interface is not None:
            self.model_interface = model_interface
        else:
            # 尝试使用 vLLM 本地模型
            if VLLMModelInterface and VLLMModelConfig:
                model_name = os.environ.get("DETECTAGENT_MODEL", "Qwen2.5-VL-7B-Instruct")
                cfg = VLLMModelConfig(model_name=model_name)
                self.model_interface = VLLMModelInterface(cfg)
            else:
                self.model_interface = None  # 无法加载本地模型时，退化为占位

    @staticmethod
    def _load_image(image_path: str) -> Image.Image:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _gen(self) -> str:
        if self.model_interface is None:
            # 无可用模型接口，占位回复（便于本地联调）
            if len(self.messages) < 3:
                return "占位回复：需要更多细节，尝试 <tool>edge_sharpening_analysis</tool>"
            return "占位回复：判断完成 <answer>yes</answer>"
        out = self.model_interface.generate([self.messages])[0]["text"]
        return out

    def start_with_image(self, image_path: str) -> str:
        img = self._load_image(image_path)
        self.messages = [
            {
                "role": "user",
                "content": first_prompt + "\n<image>",
                "multi_modal_data": {"<image>": [img]},
            }
        ]
        out = self._gen()
        self.messages.append({"role": "assistant", "content": out})
        return out

    def continue_with_tool_outputs(
        self,
        tool_output_text: str = "",
        tool_output_image_path: Optional[str] = None,
    ) -> str:
        content = ""
        multi_modal_data = None
        if tool_output_text:
            content += f"Tool Call Output: {tool_output_text}\n\n"
        if tool_output_image_path:
            # 在文本中加入一个新的 <image> 占位符，并在 multi_modal_data 中附图
            content += "<image>\n"
            multi_modal_data = {"<image>": [self._load_image(tool_output_image_path)]}
        content += continue_prompt

        user_msg: Dict = {"role": "user", "content": content}
        if multi_modal_data:
            user_msg["multi_modal_data"] = multi_modal_data
        self.messages.append(user_msg)

        out = self._gen()
        self.messages.append({"role": "assistant", "content": out})
        return out

    def final_decision(self, tool_output_text: str = "", tool_output_image_path: Optional[str] = None) -> str:
        content = ""
        multi_modal_data = None
        if tool_output_text:
            content += f"Tool Call Output: {tool_output_text}\n\n"
        if tool_output_image_path:
            content += "<image>\n"
            multi_modal_data = {"<image>": [self._load_image(tool_output_image_path)]}
        content += final_prompt

        user_msg: Dict = {"role": "user", "content": content}
        if multi_modal_data:
            user_msg["multi_modal_data"] = multi_modal_data
        self.messages.append(user_msg)

        out = self._gen()
        self.messages.append({"role": "assistant", "content": out})
        return out

    @staticmethod
    def extract_tool_tag(text: str) -> Optional[str]:
        m = re.search(r"<tool>(.*?)</tool>", text or "", re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def extract_answer_tag(text: str) -> Optional[str]:
        m = re.search(r"<answer>(.*?)</answer>", text or "", re.DOTALL)
        return m.group(1).strip() if m else None


class VAGENDetectAdapter:
    """
    面向 VAGEN 训练环境的适配器：
    - 暴露 reset(image_path) 与 step(action_name) 接口
    - 内部维护 Qwen-VL API 会话与 ToolCaller
    - 返回文本响应供上层转为观测（或自行做文本嵌入）
    """

    # 建议的动作列表（环境可复用）
    ACTIONS = [
        "gabor_texture_analysis",
        "edge_sharpening_analysis",
        "ela_analysis",
        "color_distribution_analysis",
        "final_answer",
    ]

    def __init__(self, model_interface: Optional[BaseModelInterface] = None) -> None:
        self.session = LocalVLMSession(model_interface=model_interface)
        self.tool_caller = ToolCaller()
        self.image_path: Optional[str] = None
        self.last_response: str = ""

    def reset(self, image_path: str) -> str:
        self.image_path = image_path
        self.last_response = self.session.start_with_image(image_path)
        return self.last_response

    def step(self, action_name: str) -> Tuple[str, Dict, bool]:
        """
        执行一步：
        - 若 action 为工具名：调用工具 -> 继续分析
        - 若 action 为 'final_answer'：走最终判断
        返回：(response_text, info, terminated)
        """
        if action_name == "final_answer":
            resp = self.session.final_decision()
            ans = self.session.extract_answer_tag(resp)
            self.last_response = resp
            return resp, {"answer": ans}, True

        # 工具调用
        tool_out_text = ""
        tool_out_img: Optional[str] = None
        if self.image_path:
            try:
                result = self.tool_caller.call_tool(action_name, {"image_path": self.image_path})
                tool_out_text = result.get("message", "")
                data = result.get("data") or {}
                # 常见产出 key：heatmap_path / sharpened_path / ela_path / histogram_path
                tool_out_img = (
                    data.get("heatmap_path")
                    or data.get("sharpened_path")
                    or data.get("ela_path")
                    or data.get("histogram_path")
                )
            except Exception as e:
                tool_out_text = f"tool error: {e}"

        resp = self.session.continue_with_tool_outputs(tool_out_text, tool_out_img)
        rec_tool = self.session.extract_tool_tag(resp)
        self.last_response = resp
        return resp, {"recommended_tool": rec_tool, "tool_output_image": tool_out_img}, False

    # 便捷访问器
    def get_history(self) -> List[Dict]:
        return self.session.messages

    def get_last_response(self) -> str:
        return self.last_response


# 兼容旧名称（若外部已有引用）
ImageForgeryDetectionInternalAgent = VAGENDetectAdapter

