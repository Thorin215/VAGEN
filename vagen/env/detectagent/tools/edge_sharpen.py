import cv2
import numpy as np
import os
import uuid

class EdgeSharpeningAnalyzer:
    """
    边缘锐化分析工具类
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化工具类
        参数:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def sharpen_image(self, image_path: str):
        """
        对图像进行边缘锐化
        参数:
            image_path: 输入图像路径
        返回:
            元组(结果描述文本, 锐化后图像的保存路径)
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像，请检查路径: {image_path}")

        # 创建一个锐化核
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        
        # 应用核进行锐化
        sharpened_image = cv2.filter2D(image, -1, kernel)

        # 从输入路径中获取文件名作为ID
        image_filename = os.path.basename(image_path)
        file_id = os.path.splitext(image_filename)[0]

        # 生成唯一文件名并保存图像
        sharpened_path = os.path.join(self.output_dir, f"sharpened_edge_{file_id}.png")
        cv2.imwrite(sharpened_path, sharpened_image)

        # 生成结果描述
        result_text = (
            "Edge sharpening analysis completed:\n"
            f"- Input image: {os.path.basename(image_path)}\n"
            f"- Sharpened image saved to: {sharpened_path}\n"
            f"- Sharpening can help reveal fine details and manipulation artifacts along edges."
        )

        return result_text, sharpened_path
