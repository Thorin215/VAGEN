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

def system_prompt(**kwargs):
    format_type = kwargs.get("format", "first_prompt")
    # Base system prompt for image forgery detection with available tools
    base_prompt = f"""You are an expert in image forensics. Your goal is to determine whether an image has been digitally manipulated.

Capabilities:
- Careful visual analysis (lighting, edges, color gradients, noise patterns, cloning traces)
- Iterative investigation using available tools

Tools:
<tools>
{tools_prompt_string}
</tools>

Output discipline:
- Always follow the required response format for the selected prompt type.
- Use <tool> only when you need more evidence; otherwise provide a final <answer>.
"""

    # # Add example based on format type
    # if format_type in FORMAT_CONFIGS:
    #     example = FORMAT_CONFIGS[format_type].get("example", "")
    #     if example:
    #         return base_prompt + '\n' + "Example:\n" + example
    
    return base_prompt


# Format configurations defining the structure of each format
FORMAT_CONFIGS = {
    "final_prompt": {
        "format": "<think>...</think><region>...</region><answer>...</answer>",
        "description": """
This is the final analysis of the image. Please provide your final determination based on all previous findings and analyses.
The ELA output is important for understanding the authenticity of the PNG type image. Analyze the provided forensic maps (Texture Heatmap,Edge Sharpening Map, Color Distribution Map, ELA Map) to determine whether the image has been manipulated. Think step bty step.
<think> reasoning process </think>
<region>bbox_2d : {{[x1,y1,x2,y2]}}</region>
<answer>yes/no</answer>""",
    "example": """
<think>After reviewing all available evidence including ELA and texture heatmaps, the face boundary shows inconsistent compression artifacts and sharpened edges around the eye area.</think>
<region>bbox_2d : {[120,80,240,220]}</region>
<answer>yes</answer>
""",
    },
    
    "first_prompt": {
        "format": "<think>...</think><region>...</region><answer>...</answer><tool>...</tool>",
        "description": f"""Please conduct a detailed forensic analysis to determine if this image has been digitally manipulated. Your evaluation should include:
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
    },
    
    "continue_prompt": {
        "format": "<think>...</think><region>...</region><answer>...</answer><tool>...</tool>",
        "description": f"""Based on our previous analysis and your feedback, let's continue the forensic examination of this image.
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
    }
}

def format_prompt_generator(format_type):
    """
    Generates a prompt function for the specified format type.
    
    Args:
        format_type (str): The format type to generate a prompt function for
        
    Returns:
        function: A function that generates a prompt for the specified format
    """
    def prompt_function(**kwargs):
        """
        Generate a prompt for the specified format.
            
        Returns:
            str: The formatted prompt
        """

        # Use first_prompt as default if format_type not found
        config = FORMAT_CONFIGS.get(format_type, FORMAT_CONFIGS["first_prompt"])

        base_prompt = f"{config['description']}"

        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""

        return base_prompt
    
    return prompt_function

# Generate the format prompt dictionary using the generator
format_prompt = {format_type: format_prompt_generator(format_type) 
                for format_type in FORMAT_CONFIGS}

# Convenience prompt strings for agents that expect flat prompt texts
first_prompt = format_prompt["first_prompt"]()
continue_prompt = format_prompt["continue_prompt"]()
final_prompt = format_prompt["final_prompt"]()

if __name__ == "__main__":

    for key, func in format_prompt.items():
        if key != "default":  # Skip printing default as it's the same as free_think
            print(f"{key} format prompt:")
            print(func())
            print("\n" + "="*50 + "\n")