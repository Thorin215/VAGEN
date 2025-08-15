import json

tools = [

    {
        "name": "gabor_texture_analysis",
        "description": "Analyzes the texture of an image using Gabor filters. This is useful for detecting subtle manipulations, inconsistencies in texture, and artifacts that are not easily visible. Possible Tampering Indicated by Non-Uniform Texture Distribution, pay attention to areas with high density texture patterns."
    },
    {
        "name": "edge_sharpening_analysis",
        "description": "Sharpens the edges of an image to enhance fine details. This can help reveal subtle artifacts, inconsistencies, or blurriness along object boundaries that may indicate manipulation."
    },
    {
        "name": "ela_analysis",
        "description": "Performs Error Level Analysis (ELA) to detect JPEG compression anomalies. In the ELA result, authentic areas of an image should appear dark and uniform, while manipulated regions may appear brighter and more textured."
    },
    {
        "name": "color_distribution_analysis",
        "description": "Generates a color histogram for the image. Sudden spikes, gaps, or unusual shapes in the histogram can indicate post-processing, such as contrast enhancement or color manipulation."
    }
]

tools_prompt_string = json.dumps(tools, indent=4)

continue_prompt = f"""Based on our previous analysis and your feedback, let's continue the forensic examination of this image.
Focus on these aspects for deeper verification:

New areas of interest: [Optional: Specify regions or features to re-analyze]
Cross-validate earlier findings: Check if previously detected manipulations (if any) show additional artifacts upon closer inspection.
High-precision detection: Use higher scrutiny for subtle edits like micro-liquification or gradient tampering.

You have access to the following tools to assist your analysis:
<tools>
{tools_prompt_string}
</tools>

Follow the below format for your response:
<think> reasoning process</think>
<region>bbox_2d : {{[x1,y1,x2,y2]}}</region>
<answer>yes/no/unsure(if you want to use tool for further research)</answer>
<tool>If you need to use a tool, specify only its name here</tool>
"""

first_prompt = f"""Please conduct a detailed forensic analysis to determine if this image has been digitally manipulated. Your evaluation should include:
Comprehensive analysis of potential manipulation signs, such as:

Inconsistencies in lighting/shadows
Irregular edges or artifacts around objects
Unnatural color gradients or noise patterns
Signs of cloning, healing, or inpainting

If manipulation is detected, specify the exact type(s) of edits from these categories:
- Attribute Enhancement (e.g., facial retouching, body reshaping)
- Color Modification (e.g., saturation, contrast adjustments)
- Object Addition (unnatural elements inserted)
- Object Removal (evidence of erasure or patching)
- Object Replacement (one object swapped for another)

You have access to the following tools to assist your analysis:
<tools>
{tools_prompt_string}
</tools>

Follow the below format for your response:

<think> Reasoning Process </think>
<region>bbox_2d : {{[x1,y1,x2,y2]}}</region>
<answer>yes/no/unsure(if you want to use tool for further research)</answer>
<tool>If you need to use a tool, specify only its name here</tool>"""

final_prompt = """
This is the final analysis of the image. Please provide your final determination based on all previous findings and analyses.
The ELA output is important for understanding the authenticity of the PNG type image. Analyze the provided forensic maps (Texture Heatmap,Edge Sharpening Map, Color Distribution Map, ELA Map) to determine whether the image has been manipulated. Think step bty step.
<think> reasoning process </think>
<region>bbox_2d : {{[x1,y1,x2,y2]}}</region>
<answer>yes/no</answer>
<tool></tool>
"""