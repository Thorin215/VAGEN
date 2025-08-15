import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple
import uuid

class GaborTextureAnalyzer:
    """
    Gabor纹理分析工具类，用于生成纹理热度图
    """
    
    def __init__(self, output_dir: str = "/tool/output"):
        """
        初始化工具类
        参数:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _apply_gabor_filter(self, image: np.ndarray, ksize: int = 31, sigma: float = 5, 
                          theta: float = 0, lambd: float = 10, gamma: float = 0.5, 
                          psi: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用Gabor滤波器(内部方法)
        参数:
            image: 输入灰度图像
            ksize: 核大小(奇数)
            sigma: 高斯函数的标准差
            theta: 滤波器的方向(弧度)
            lambd: 正弦函数的波长
            gamma: 空间纵横比
            psi: 相位偏移
        返回:
            滤波后的图像和核
        """
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        return filtered, kernel
    
    def _gabor_filter_bank(self, image: np.ndarray, orientations: int = 8, scales: int = 4) -> Tuple[list, np.ndarray]:
        """
        Gabor滤波器组应用(内部方法)
        参数:
            image: 输入灰度图像
            orientations: 方向数量
            scales: 尺度数量
        返回:
            滤波结果列表和能量特征图
        """
        ksize = 31
        sigma = 5
        lambd = 10
        gamma = 0.5
        psi = 0
        
        results = []
        energy_maps = []
        
        for scale in range(1, scales+1):
            for orient in range(orientations):
                theta = orient * np.pi / orientations
                current_lambd = lambd * scale
                
                filtered, _ = self._apply_gabor_filter(
                    image, ksize, sigma, theta, current_lambd, gamma, psi
                )
                
                energy = filtered ** 2
                energy_maps.append(energy)
                results.append(filtered)
        
        combined_energy = np.mean(np.array(energy_maps), axis=0)
        return results, combined_energy
    
    def generate_texture_heatmap(self, image_path: str, orientations: int = 8, scales: int = 4) -> Tuple[str, str]:
        """
        生成纹理热度图
        参数:
            image_path: 输入图像路径
            orientations: 方向数量
            scales: 尺度数量
        返回:
            元组(结果描述文本, 热度图保存路径)
        """
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法加载图像，请检查路径: {image_path}")
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 应用Gabor滤波器组
        _, energy_map = self._gabor_filter_bank(image, orientations, scales)
        
        # 归一化能量图用于显示
        energy_map_normalized = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX)
        energy_map_normalized = energy_map_normalized.astype(np.uint8)
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(energy_map_normalized, cv2.COLORMAP_JET)
        
        # 从输入路径中获取文件名作为ID
        image_filename = os.path.basename(image_path)
        file_id = os.path.splitext(image_filename)[0]
        
        # 生成唯一文件名
        heatmap_path = os.path.join(self.output_dir, f"texture_heatmap_{file_id}.png")
        
        # 保存热度图
        cv2.imwrite(heatmap_path, heatmap)
        
        # 生成结果描述
        result_text = (
            "Texture analysis completed:\n"
            f"- Input image: {os.path.basename(image_path)}\n"
            # f"- Number of orientations: {orientations}\n"
            # f"- Number of scales: {scales}\n"
            f"- Heatmap saved to: {heatmap_path}\n"
            f"- High-intensity regions indicate areas with rich texture"
        )
        
        return result_text, heatmap_path
    
    def visualize_results(self, image_path: str, orientations: int = 8, scales: int = 4) -> str:
        """
        将原始图像与纹理热度图并排可视化
        参数:
            image_path: 输入图像路径
            orientations: 方向数量
            scales: 尺度数量
        返回:
            可视化结果保存路径
        """
        # 读取原始图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法加载图像，请检查路径: {image_path}")

        # 转换为灰度图进行分析
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # 归一化
        gray_image_normalized = gray_image.astype(np.float32) / 255.0
        
        # 应用Gabor滤波器组获取能量图
        _, energy_map = self._gabor_filter_bank(gray_image_normalized, orientations, scales)
        
        # 归一化能量图并应用颜色映射生成热度图
        energy_map_normalized = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(energy_map_normalized, cv2.COLORMAP_JET)
        
        # 将OpenCV的BGR图像转换为matplotlib的RGB格式
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # 创建并排显示的可视化图
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 显示原始图像
        axes[0].imshow(original_image_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 显示热度图
        axes[1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)) # BGR to RGB for heatmap
        axes[1].set_title('Texture Heatmap')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # 保存可视化结果
        unique_id = uuid.uuid4().hex
        visualization_path = os.path.join(self.output_dir, f"gabor_visualization_{unique_id}.png")
        plt.savefig(visualization_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        return visualization_path


# 使用示例
if __name__ == "__main__":
    # 创建纹理分析工具实例
    analyzer = GaborTextureAnalyzer()
    
    # 示例图像路径 - 请替换为实际路径
    test_image_path = "/mnt/d/86134/Desktop/M4Agent/test.png"
    
    try:
        # 生成纹理热度图
        result_text, heatmap_path = analyzer.generate_texture_heatmap(test_image_path)
        print(result_text)
        print(f"热度图已保存至: {heatmap_path}")
        
        # 可选: 生成可视化结果
        visualization_path = analyzer.visualize_results(test_image_path)
        print(f"可视化结果已保存至: {visualization_path}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")