import os
import uuid
from PIL import Image, ImageChops, ImageEnhance

class ELAAnalyzer:
    """
    Error Level Analysis (ELA) a tool for detecting JPEG compression anomalies.
    """

    def __init__(self, output_dir: str = "output"):
        """
        Initializes the ELA analyzer.
        Args:
            output_dir: The directory to save output images to.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze(self, image_path: str, quality: int = 90, scale: int = 15):
        """
        Performs Error Level Analysis on a given image.
        Args:
            image_path: The path to the input image.
            quality: The quality level for JPEG re-compression (1-100).
            scale: The enhancement scale for the final ELA image.
        Returns:
            A tuple containing a descriptive text and the path to the saved ELA image.
        """
        try:
            original_image = Image.open(image_path).convert('RGB')
            
            # Create a temporary path for the re-saved image
            temp_resaved_path = os.path.join(self.output_dir, f"temp_{uuid.uuid4().hex}.jpg")
            
            # Re-save the image at a specific quality
            original_image.save(temp_resaved_path, 'JPEG', quality=quality)
            resaved_image = Image.open(temp_resaved_path)
            
            # Calculate the difference between the original and re-saved images
            ela_image = ImageChops.difference(original_image, resaved_image)
            
            # Enhance the ELA image to make the differences more visible
            enhancer = ImageEnhance.Brightness(ela_image)
            ela_image = enhancer.enhance(scale)
            
            # Clean up the temporary re-saved file
            os.remove(temp_resaved_path)
            
            # 从输入路径中获取文件名作为ID
            image_filename = os.path.basename(image_path)
            file_id = os.path.splitext(image_filename)[0]

            # Save the final ELA result image
            ela_path = os.path.join(self.output_dir, f"ela_result_{file_id}.png")
            ela_image.save(ela_path)

            result_text = (
                "ELA (Error Level Analysis) completed:\n"
                f"- Input image: {os.path.basename(image_path)}\n"
                f"- ELA image saved to: {ela_path}\n"
                "In the ELA result, authentic areas of an image should appear dark and uniform, "
                "while manipulated regions may appear brighter and more textured."
            )

            return result_text, ela_path

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_resaved_path):
                os.remove(temp_resaved_path)
            raise e
