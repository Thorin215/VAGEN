import os
from typing import Dict, Any
import uuid
from .tools.gabor import GaborTextureAnalyzer
from .tools.edge_sharpen import EdgeSharpeningAnalyzer
from .tools.ela import ELAAnalyzer
from .tools.color_distribution import ColorDistributionAnalyzer

class ToolCaller:
    """
    工具调用器，用于统一调用各种图像处理工具
    """
    
    def __init__(self, output_dir: str = "tool_output"):
        """
        初始化工具调用器
        参数:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化工具实例
        self.gabor_analyzer = GaborTextureAnalyzer(os.path.join(output_dir, "gabor_results"))
        self.edge_analyzer = EdgeSharpeningAnalyzer(os.path.join(output_dir, "edge_results"))
        self.ela_analyzer = ELAAnalyzer(os.path.join(output_dir, "ela_results"))
        self.color_analyzer = ColorDistributionAnalyzer(os.path.join(output_dir, "color_results"))
    
    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用指定工具
        参数:
            tool_name: 工具名称 ('gabor_texture_analysis', 'edge_sharpening_analysis', 'ela_analysis', 'color_distribution_analysis')
            params: 工具参数字典
        返回:
            结果字典，包含:
            - status: 执行状态 ('success'/'error')
            - message: 结果消息
            - data: 结果数据(如图像路径)
        """
        # 生成唯一请求ID
        request_id = uuid.uuid4().hex
        
        try:
            if tool_name == "gabor_texture_analysis":
                return self._call_gabor_analyzer(params, request_id)
            elif tool_name == "edge_sharpening_analysis":
                return self._call_edge_analyzer(params, request_id)
            elif tool_name == "ela_analysis":
                return self._call_ela_analyzer(params, request_id)
            elif tool_name == "color_distribution_analysis":
                return self._call_color_distribution_analyzer(params, request_id)
            else:
                return {
                    "status": "error",
                    "message": f"Undefined Tool Name: {tool_name} (please call only one tool at a time)",
                    "data": None
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"工具调用失败: {str(e)}",
                "data": None
            }

    def _call_ela_analyzer(self, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """
        调用ELA分析工具(内部方法)
        """
        if "image_path" not in params:
            raise ValueError("缺少必要参数 'image_path'")
        
        result_text, ela_path = self.ela_analyzer.analyze(
            image_path=params["image_path"]
        )
        
        result_data = {
            "ela_path": ela_path,
            "request_id": request_id
        }
        
        return {
            "status": "success",
            "message": result_text,
            "data": result_data
        }

    def _call_color_distribution_analyzer(self, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """
        调用色彩分布分析工具(内部方法)
        """
        if "image_path" not in params:
            raise ValueError("缺少必要参数 'image_path'")
        
        result_text, histogram_path = self.color_analyzer.analyze(
            image_path=params["image_path"]
        )
        
        result_data = {
            "histogram_path": histogram_path,
            "request_id": request_id
        }
        
        return {
            "status": "success",
            "message": result_text,
            "data": result_data
        }

    def _call_edge_analyzer(self, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """
        调用边缘锐化分析工具(内部方法)
        """
        if "image_path" not in params:
            raise ValueError("缺少必要参数 'image_path'")
        
        result_text, sharpened_path = self.edge_analyzer.sharpen_image(
            image_path=params["image_path"]
        )
        
        result_data = {
            "sharpened_path": sharpened_path,
            "request_id": request_id
        }
        
        return {
            "status": "success",
            "message": result_text,
            "data": result_data
        }

    def _call_gabor_analyzer(self, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """
        调用Gabor纹理分析工具(内部方法)
        参数:
            params: 必须包含 'image_path'，可选包含 'orientations' 和 'scales'
            request_id: 请求ID
        返回:
            结果字典
        """
        # 参数检查
        if "image_path" not in params:
            raise ValueError("缺少必要参数 'image_path'")
        
        # 设置默认参数
        orientations = params.get("orientations", 8)
        scales = params.get("scales", 4)
        
        # 调用工具
        result_text, heatmap_path = self.gabor_analyzer.generate_texture_heatmap(
            image_path=params["image_path"],
            orientations=orientations,
            scales=scales
        )
        
        # 可选: 生成可视化结果
        visualization_path = None
        if params.get("generate_visualization", False):
            visualization_path = self.gabor_analyzer.visualize_results(
                params["image_path"],
                orientations=orientations,
                scales=scales
            )
        
        # 构建返回结果
        result_data = {
            "heatmap_path": heatmap_path,
            "visualization_path": visualization_path,
            "request_id": request_id
        }
        
        return {
            "status": "success",
            "message": result_text,
            "data": result_data
        }


# 使用示例
if __name__ == "__main__":
    # 创建工具调用器
    tool_caller = ToolCaller()
    
    # 准备参数
    image_path = "/mnt/d/86134/Desktop/M4Agent/test.png"
    
    # 调用Gabor工具
    gabor_params = {
        "image_path": image_path,
        "orientations": 8,
        "scales": 4,
        "generate_visualization": True
    }
    gabor_result = tool_caller.call_tool("gabor_texture_analysis", gabor_params)
    if gabor_result["status"] == "success":
        print("Gabor工具调用成功:")
        print(gabor_result["message"])
    else:
        print("Gabor工具调用失败:", gabor_result["message"])

    # 调用边缘锐化工具
    edge_params = {"image_path": image_path}
    edge_result = tool_caller.call_tool("edge_sharpening_analysis", edge_params)
    if edge_result["status"] == "success":
        print("\n边缘锐化工具调用成功:")
        print(edge_result["message"])
    else:
        print("\n边缘锐化工具调用失败:", edge_result["message"])

    # 调用ELA工具
    ela_params = {"image_path": image_path}
    ela_result = tool_caller.call_tool("ela_analysis", ela_params)
    if ela_result["status"] == "success":
        print("\nELA工具调用成功:")
        print(ela_result["message"])
    else:
        print("\nELA工具调用失败:", ela_result["message"])

    # 调用色彩分布工具
    color_params = {"image_path": image_path}
    color_result = tool_caller.call_tool("color_distribution_analysis", color_params)
    if color_result["status"] == "success":
        print("\n色彩分布工具调用成功:")
        print(color_result["message"])
    else:
        print("\n色彩分布工具调用失败:", color_result["message"])

