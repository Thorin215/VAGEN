import os
from PIL import Image
import numpy as np

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_image(image_array, original_path, suffix):
    """
    保存处理后的图像。
    
    Args:
        image_array (np.ndarray): 图像数据。
        original_path (str): 原始图像路径，用于生成新文件名。
        suffix (str): 添加到新文件名的后缀 (e.g., 'sharpened', 'ela').

    Returns:
        str: 保存后的图像路径。
    """
    try:
        base, ext = os.path.splitext(original_path)
        output_path = f"{base}_{suffix}{ext}"
        ensure_dir(output_path)
        
        # 从 numpy 数组创建 PIL Image 对象
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(image_array)
        
        img.save(output_path)
        print(f"[Utils] 图像已保存到: {output_path}")
        return output_path
    except Exception as e:
        print(f"[Utils] 保存图像时出错: {e}")
        return None
