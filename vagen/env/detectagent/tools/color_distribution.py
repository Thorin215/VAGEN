import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid

class ColorDistributionAnalyzer:
    """
    Analyzes the color distribution of an image by generating histograms for each color channel.
    """

    def __init__(self, output_dir: str = "output"):
        """
        Initializes the color distribution analyzer.
        Args:
            output_dir: The directory to save output images to.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze(self, image_path: str):
        """
        Generates and saves a color histogram for the given image.
        Args:
            image_path: The path to the input image.
        Returns:
            A tuple containing a descriptive text and the path to the saved histogram image.
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from path: {image_path}")

        # Split the image into its B, G, R channels
        channels = cv2.split(image)
        colors = ('b', 'g', 'r')
        
        plt.figure(figsize=(10, 6))
        plt.title('Color Channel Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Number of Pixels')

        # Calculate and plot the histogram for each channel
        for (channel, color) in zip(channels, colors):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        
        plt.legend(['Blue Channel', 'Green Channel', 'Red Channel'])
        plt.grid(True)

        # 从输入路径中获取文件名作为ID
        image_filename = os.path.basename(image_path)
        file_id = os.path.splitext(image_filename)[0]

        # Save the histogram plot
        histogram_path = os.path.join(self.output_dir, f"color_histogram_{file_id}.png")
        plt.savefig(histogram_path)
        plt.close()

        result_text = (
            "Color distribution analysis completed:\n"
            f"- Input image: {os.path.basename(image_path)}\n"
            f"- Color histogram saved to: {histogram_path}\n"
            "Sudden spikes, gaps, or unusual shapes in the histogram can indicate "
            "post-processing, such as contrast enhancement or color manipulation."
        )

        return result_text, histogram_path
